#define FMT_HEADER_ONLY
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#define SPDLOG_FMT_EXTERNAL
#include <getopt.h>
#include <spdlog/spdlog.h>

#include <asio.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <chrono>
#include <map>

using spdlog::error;
using spdlog::info;
using spdlog::warn;

using asio::awaitable;
using asio::buffer;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::ip::tcp;
using std::chrono::steady_clock;

using namespace std::chrono_literals;
using namespace asio::experimental::awaitable_operators;
constexpr auto use_nothrow_awaitable = asio::as_tuple(use_awaitable);

namespace {
/////////////////////////////////////////////////////////////////////////////
// MIMP Global Config/Stats
std::atomic_uint64_t active_conn{0};
steady_clock::duration relay_timeout{0s};
steady_clock::duration connect_timeout{5s};

// The pair of valid users
std::map<std::string, std::string> auth_table;

// The transferred bytes in the last cycle
std::atomic_uint64_t up_bytes{0};
std::atomic_uint64_t down_bytes{0};
bool is_last_non_zero{true};

/////////////////////////////////////////////////////////////////////////////
// timeout utils

/// \brief Return if `deadline` has passed and watchdog is awake, which might
/// short-circuit other operations.
awaitable<void>
watchdog(steady_clock::time_point &deadline, std::function<void()> callback = nullptr) noexcept {
    asio::steady_timer timer{co_await asio::this_coro::executor};
    auto now = steady_clock::now();

    // If watchdog is awake and the deadline is reached, return.
    while (now < deadline) {
        timer.expires_at(deadline);
        co_await timer.async_wait(use_nothrow_awaitable);
        now = steady_clock::now();
    }

    if (callback)
        callback();
}

/// \brief Stop the execution of the coroutine after `timeout`.
///
/// \remark This implementation is somehow inefficient but convenient, and is
/// intended to be used in case where executing `a` takes a rather long time.
template <typename T, typename AwaitableExecutor>
awaitable<std::optional<T>> stop_after(
    steady_clock::duration timeout, awaitable<T, AwaitableExecutor> a,
    std::function<void()> callback = nullptr
) {
    asio::steady_timer timer{co_await asio::this_coro::executor};
    timer.expires_after(timeout);

    auto ret = co_await (std::move(a) || timer.async_wait(use_nothrow_awaitable));
    if (ret.index() == 0) {
        co_return std::get<0>(ret);
    } else {
        if (callback)
            callback();
        co_return std::nullopt;
    }
}

/////////////////////////////////////////////////////////////////////////////
// socks5 utils

/// \brief Validate the authentication of the socks5 request.
awaitable<bool> validate_auth(tcp::socket &client) noexcept {
    try {
        // It begins with the client producing a Username/Password request:
        //         +----+------+----------+------+----------+
        //         |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
        //         +----+------+----------+------+----------+
        //         | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
        //         +----+------+----------+------+----------+
        uint8_t ver{0};
        uint8_t ulen{0};
        co_await asio::async_read(client, buffer(&ver, 1), use_awaitable);
        co_await asio::async_read(client, buffer(&ulen, 1), use_awaitable);

        std::string uname(ulen, '\0');
        co_await asio::async_read(client, buffer(uname), use_awaitable);

        uint8_t plen{0};
        co_await asio::async_read(client, buffer(&plen, 1), use_awaitable);

        std::string passwd(plen, '\0');
        co_await asio::async_read(client, buffer(passwd), use_awaitable);

        // Check the authentication table
        if (auth_table.contains(uname) && auth_table[uname] == passwd)
            co_return true;
        else
            co_return false;
    } catch (std::exception const &e) { error("unresolved socks5 auth exception: {}", e.what()); }

    co_return false;
}

/// \brief Respond to the socks5 authentication asynchornously
awaitable<void> reply_auth(uint8_t reply_type, tcp::socket &client) {
    co_await asio::async_write(client, buffer<uint8_t>({0x05, reply_type}), use_awaitable);
}

/// \brief Parse the ATYP field of the socks5 request
awaitable<std::optional<tcp::resolver::query>>
parse_atyp(uint8_t atyp_type, tcp::socket &client) noexcept {
    // Parse the ATYP field
    // In an address field (DST.ADDR, BND.ADDR), the ATYP field specifies the
    // type of address contained within the field as follows:
    try {
        switch (atyp_type) {
        default: error("invalid socks5 address type: {}", atyp_type); co_return std::nullopt;

        case 0x01: {
            // the address is a version-4 IP address, with a length of 4 octets
            uint16_t port{0};
            asio::detail::array<uint8_t, 4> ipv4_address;
            co_await asio::async_read(client, buffer(ipv4_address), use_awaitable);
            co_await asio::async_read(client, buffer(&port, 2), use_awaitable);

            port = ntohs(port);
            co_return tcp::resolver::query{
                /*host=    */ asio::ip::make_address_v4(ipv4_address).to_string(),
                /*service= */ std::to_string(port)
            };
        }

        case 0x03: {
            // the address field contains a fully-qualified domain name. The first
            // octet of the address field contains the number of octets of name that
            // follow, there is no terminating NUL octet.
            uint8_t domain_length{0};
            co_await asio::async_read(client, buffer(&domain_length, 1), use_awaitable);

            uint16_t port{0};
            std::string domain_name(domain_length, '\0');
            co_await asio::async_read(client, buffer(domain_name), use_awaitable);
            co_await asio::async_read(client, buffer(&port, 2), use_awaitable);

            port = ntohs(port);
            co_return tcp::resolver::query{
                /*host=    */ domain_name,
                /*service= */ std::to_string(port)
            };
        }

        case 0x04: {
            // the address is a version-6 IP address, with a length of 16 octets.
            uint16_t port{0};
            asio::detail::array<uint8_t, 16> ipv6_address;
            co_await asio::async_read(client, buffer(ipv6_address), use_awaitable);
            co_await asio::async_read(client, buffer(&port, 2), use_awaitable);

            port = ntohs(port);
            co_return tcp::resolver::query{
                /*host=    */ asio::ip::make_address_v6(ipv6_address).to_string(),
                /*service= */ std::to_string(port)
            };
            break;
        }
        }
    } catch (std::exception &e) {
        // do nothing
        error("unresolved ATYP parsing exception: {}", e.what());
        co_return std::nullopt;
    }
}

/// \brief Respond to the socks5 request asynchornously
awaitable<void> reply_conn_req(uint8_t reply_type, tcp::socket &client) {
    co_await asio::async_write(
        client,
        buffer<uint8_t>({
            0x05,                   // protocol version: X'05'
            reply_type,             // custom reply type
            0x00,                   // RESERVED
            0x01,                   // address type of following address
            0x00, 0x00, 0x00, 0x00, // server bound address
            0x00, 0x00              // server bound port in network octet order
        }),
        use_awaitable
    );
}

/// \brief Copy the data from the source to the destination
awaitable<void> copy_directional(
    tcp::socket &to, tcp::socket &from, steady_clock::time_point &deadline,
    std::function<void(size_t)> callback = nullptr
) noexcept {
    std::array<uint8_t, 1024> buf;
    for (;;) {
        // Update the deadline to indicate that this copy is active
        deadline = std::max(deadline, steady_clock::now() + relay_timeout);

        auto [e1, n1] = co_await from.async_read_some(buffer(buf), use_nothrow_awaitable);
        if (e1)
            break;

        auto [e2, n2] = co_await asio::async_write(to, buffer(buf, n1), use_nothrow_awaitable);
        if (e2)
            break;

        // Invoke the callback by the number of bytes copied
        if (callback)
            callback(n1);
    }
}

/// \brief Copy the data bidirectionally between the client and the server
awaitable<void> copy_bidirectional(tcp::socket &client, tcp::socket &server) noexcept {
    steady_clock::time_point client_to_server_deadline{steady_clock::now() + relay_timeout};
    steady_clock::time_point server_to_client_deadline{steady_clock::now() + relay_timeout};

    active_conn.fetch_add(1);

    auto update_up_bytes = [&](size_t n1) { up_bytes.fetch_add(n1); };
    auto update_down_bytes = [&](size_t n1) { down_bytes.fetch_add(n1); };

    // If timeout is set to zero, disable watchdog
    if (relay_timeout == 0s) {
        co_await (
            copy_directional(client, server, server_to_client_deadline, update_down_bytes) &&
            copy_directional(server, client, client_to_server_deadline, update_up_bytes)
        );
    } else {
        co_await (
            (copy_directional(client, server, server_to_client_deadline, update_down_bytes) ||
             watchdog(
                 server_to_client_deadline,
                 [] { warn("relay is timeout-ed after {} s", relay_timeout / 1.0s); }
             )) &&
            (copy_directional(server, client, client_to_server_deadline, update_down_bytes) ||
             watchdog(client_to_server_deadline))
        );
    }

    active_conn.fetch_sub(1); // noexcept and safe
}

/// \brief Receive and parse the socks5 request asynchornously
awaitable<void> handle_socks5(tcp::socket client) {
    try {
        std::array<uint8_t, 1024> buf;
        co_await asio::async_read(client, buffer(buf, 2), use_awaitable);

        ///////////////////////////////////////////////////////////////////////////
        // The client connects to the server, and sends a version identifier/method
        // selection message:
        //           +----+----------+----------+
        //           |VER | NMETHODS | METHODS  |
        //           +----+----------+----------+
        //           | 1  |    1     | 1 to 255 |
        //           +----+----------+----------+
        if (buf[0] != 0x05) {
            error("invalid socks5 version: {}", buf[0]);
            co_return;
        }

        uint8_t const nmethods = buf[1];
        co_await asio::async_read(client, buffer(buf, nmethods), use_awaitable);

        ///////////////////////////////////////////////////////////////////////////
        // The server selects from one of the methods given in METHODS, an sends a
        // METHOD selection message:
        //           +----+--------+
        //           |VER | METHOD |
        //           +----+--------+
        //           | 1  |   1    |
        //           +----+--------+
        //  o  X'00' NO AUTHENTICATION REQUIRED
        //  o  X'01' GSSAPI
        //  o  X'02' USERNAME/PASSWORD
        //  o  X'03' to X'7F' IANA ASSIGNED
        //  o  X'80' to X'FE' RESERVED FOR PRIVATE METHODS
        //  o  X'FF' NO ACCEPTABLE METHODS
        bool const auth_required = !auth_table.empty();

        bool noauth_supported = false;
        bool auth_supported = false;
        for (size_t i = 0; i < nmethods; ++i)
            if (buf[i] == 0x00)
                noauth_supported = true;
            else if (buf[i] == 0x02)
                auth_supported = true;

        auto reply_method_selection = reply_auth;
        auto auth_and_reply = [&]() -> awaitable<bool> {
            // When authentication is selected, enter the sub-negotiation.
            auto succeed = co_await validate_auth(client);

            ///////////////////////////////////////////////////////////////////////////
            // The server verifies the supplied UNAME and PASSWD, and sends the
            //    following response:
            //
            //        +----+--------+
            //        |VER | STATUS |
            //        +----+--------+
            //        | 1  |   1    |
            //        +----+--------+
            //  A STATUS field of X'00' indicates success. If the server returns a
            //  `failure' (STATUS value other than X'00') status, it MUST close the
            //  connection.
            if (succeed) {
                co_await reply_auth(0x00, client);
                co_return false;
            } else {
                co_await reply_auth(0xFF, client);
                co_return true;
            }
        };

        if (auth_required) {
            if (!auth_supported) {
                warn("authentication is required, but is not supported by the client "
                     "side");
                co_await reply_method_selection(0xFF, client);
                co_return;
            }

            // auth supported
            co_await reply_method_selection(0x02, client);

            // Enter the sub-negotiation
            bool const should_terminate = co_await auth_and_reply();
            if (should_terminate) {
                warn(
                    "authentication failed at {}:{}",
                    client.remote_endpoint().address().to_string(), client.remote_endpoint().port()
                );
                co_return;
            }
        } else /* auth not required */
            if (noauth_supported) {
                // authentication is not required, and no authentication is supported
                co_await reply_method_selection(0x00, client);
            } else if (auth_supported) {
                // Deal with the weird case that auth is not required, but only auth is
                // supported.
                bool const should_terminate = co_await auth_and_reply();
                if (should_terminate) {
                    warn(
                        "authentication failed at {}:{}",
                        client.remote_endpoint().address().to_string(),
                        client.remote_endpoint().port()
                    );
                    co_return;
                }
            } else {
                warn("no acceptable socks5 method");
                co_await reply_method_selection(0xFF, client);
                co_return;
            }

        ///////////////////////////////////////////////////////////////////////////
        // The client and server then enter a method-specific sub-negotiation.
        // The SOCKS request is formed as follows:
        //    +----+-----+-------+------+----------+----------+
        //    |VER | CMD |  RSV  | ATYP | DST.ADDR | DST.PORT |
        //    +----+-----+-------+------+----------+----------+
        //    | 1  |  1  | X'00' |  1   | Variable |    2     |
        //    +----+-----+-------+------+----------+----------+
        // o  VER    protocol version: X'05'
        // o  CMD
        //    o  CONNECT X'01'
        //    o  BIND X'02'
        //    o  UDP ASSOCIATE X'03'
        // o  RSV    RESERVED
        // o  ATYP   address type of following address
        //    o  IP V4 address: X'01'
        //    o  DOMAINNAME: X'03'
        //    o  IP V6 address: X'04'
        // o  DST.ADDR       desired destination address
        // o  DST.PORT desired destination port in network octet
        //    order
        co_await asio::async_read(client, buffer(buf, 4), use_awaitable);
        if (buf[0] != 0x05) {
            error("invalid socks5 version: {}", buf[0]);
            co_return;
        }

        bool has_command = true;
        if (buf[1] != 0x01) {
            // handled later
            has_command = false;
        }

        if (buf[2] != 0x00) {
            error("invalid socks5 reserved: {}, should be 0x00", buf[2]);
            co_return;
        }

        auto query = co_await parse_atyp(buf[3], client);

        ///////////////////////////////////////////////////////////////////////////
        // The SOCKS request information is sent by the client as soon as it has
        // established a connection to the SOCKS server, and completed the
        // authentication negotiations.  The server evaluates the request, and
        // returns a reply formed as follows:
        //      +----+-----+-------+------+----------+----------+
        //      |VER | REP |  RSV  | ATYP | BND.ADDR | BND.PORT |
        //      +----+-----+-------+------+----------+----------+
        //      | 1  |  1  | X'00' |  1   | Variable |    2     |
        //      +----+-----+-------+------+----------+----------+
        if (!has_command) {
            warn("invalid socks5 command: {}, only CONNECT(0x01) is supported for now", buf[1]);
            co_await reply_conn_req(0x07 /* command not supported */, client);
            co_return;
        }

        if (!query.has_value()) {
            warn("failed to parse socks5 ATYP field");
            co_await reply_conn_req(0x01 /* general SOCKS server failure */, client);
            co_return;
        }

        // Build the connection to the target server and copy bidirectionally
        assert(query.has_value());
        tcp::socket server(client.get_executor());
        tcp::resolver resolver(client.get_executor());
        {
            auto op = co_await stop_after(
                connect_timeout,
                asio::async_connect(server, resolver.resolve(query.value()), use_nothrow_awaitable),
                [&] {
                warn(
                    "connecting to {}:{} timeout after 5s", query->host_name(),
                    query->service_name()
                );
            }
            );
            if (!op.has_value()) {
                co_await reply_conn_req(0x04 /* Host unreachable */, client);
                co_return;
            }

            auto [e, endpoint] = op.value();
            if (e) {
                warn(
                    "failed to get connected to the server at {}:{}", query->host_name(),
                    query->service_name()
                );
                co_await reply_conn_req(0x05 /* connection refused */, client);
                co_return;
            }
        }

        // Respond to the client that the connection is established
        co_await reply_conn_req(0x00 /* succeeded */, client);

        // Actually execute the bi-directional copy
        info("relay to {}:{} is established", query->host_name(), query->service_name());

        co_await copy_bidirectional(client, server);

        info(
            "relay to {}:{} is closed, lasting {} connections active", query->host_name(),
            query->service_name(), active_conn.load()
        );
    } catch (asio::system_error const &e) {
        // Ignore EOF exception
        if (e.code() != asio::error::eof)
            warn("unresolved socks5 handler asio exception: {}", e.code().message());
    } catch (std::exception const &e) {
        warn("unresolved socks5 handler local exception: {}", e.what());
    }
}

awaitable<void> dispatch_connection(tcp::socket request) {
    co_await handle_socks5(std::move(request));
}

awaitable<void> listener(tcp::acceptor acceptor) {
    auto local_endpoint = acceptor.local_endpoint();
    info("listening on {}:{}", local_endpoint.address().to_string(), local_endpoint.port());
    for (;;) {
        tcp::socket socket = co_await acceptor.async_accept(use_awaitable);
        auto executor = acceptor.get_executor();
        co_spawn(executor, dispatch_connection(std::move(socket)), detached);
    }
}

awaitable<void> print_bandwidth() {
    asio::steady_timer timer{co_await asio::this_coro::executor};
    constexpr auto duration = 1s;

    auto format_bytes = [](size_t bytes) -> std::string {
        double bps = static_cast<double>(bytes * 8);

        // Determine the appropriate unit
        constexpr std::array units = {"bps", "Kbps", "Mbps", "Gbps", "Tbps"};
        int unit_index = 0;
        while (bps >= 1000.0 && unit_index < 4) {
            bps /= 1000.0;
            ++unit_index;
        }

        return fmt::format("{:.2f} {}", bps, units[unit_index]);
    };

    for (;;) {
        timer.expires_after(duration);
        co_await timer.async_wait(use_nothrow_awaitable);

        // print the statistics
        auto const up_bytes_ = up_bytes.load();
        auto const down_bytes_ = down_bytes.load();
        up_bytes.store(0);
        down_bytes.store(0);

        if (up_bytes_ != 0 || down_bytes_ != 0) {
            info("Up: {} / Down: {}", format_bytes(up_bytes_), format_bytes(down_bytes_));
            is_last_non_zero = true;
        } else if (is_last_non_zero) {
            info("Up: {} / Down: {}", format_bytes(up_bytes_), format_bytes(down_bytes_));
            is_last_non_zero = false;
        }
    }
}
} // namespace

int main(int argc, char *argv[]) {
    try {
        int opt;
        int port = 23333;

        auto print_help = [&]() {
            // clang-format off
      std::ostringstream oss;
      oss << "mimp 0.1.0, the man in the middle proxy\n"
          << fmt::format("Usage: {} [options]\n", argv[0])
          << fmt::format("Options: \n")
          << fmt::format("  -h, --help               print this help text\n")
          << fmt::format("  -p, --port=PORT          the port to listen to (23333 by default)\n")
          << fmt::format("  -t, --timeout=TIME       the timeout delay of a relay in ms (disabled by default)\n")
          << fmt::format("                           set to 0 to disable timeout\n")
          << fmt::format("  -a, --auth=UNAME,PASSWD  add users to enable authentication\n")
          << fmt::format("Example usages: \n")
          << fmt::format("  mimp -p 23333 -t 2000    set the port to 23333 and timeout after 2s\n")
          << fmt::format("  mimp -a f1,b1 -a f2,b2   enable auth and allow two pairs of U/P\n");
            // clang-format on
            fmt::print("{}", oss.str());
        };

        // Define long options
        static option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"port", required_argument, 0, 'p'},
            {"timeout", required_argument, 0, 't'},
            {"auth", required_argument, 0, 'a'},
            {0, 0, 0, 0}
        };

        // Parse terminal parameters
        while ((opt = getopt_long(argc, argv, "hp:t:a:", long_options, nullptr)) != -1) {
            switch (opt) {
            case 'h': print_help(); return 0;
            case 'p':
                port = std::stoi(optarg);
                if (port < 0 || port > 65535)
                    throw std::out_of_range("invalid port specification");
                break;
            case 't':
                if (optarg[0] == '-')
                    throw std::out_of_range("invalid timeout specification");
                else
                    relay_timeout = std::stoul(optarg) * 1ms;
                break;
            case 'a': {
                auto const oarg = std::string(optarg);
                if (oarg.find(',') == std::string::npos)
                    throw std::invalid_argument("invalid auth specification");
                else {
                    auto pos = oarg.find(',');
                    auth_table[oarg.substr(0, pos)] = oarg.substr(pos + 1);
                }
                break;
            }
            case '?':
            default: break;
            }
        }

        // Check
        if (relay_timeout == 0s)
            info("timeout is disabled");
        else
            info("timeout is set to be {} s", relay_timeout / 1.0s);

        if (auth_table.empty())
            info("authentication is not required");
        else
            info("authentication is required");

        // Create the I/O context that will run the coroutine
        asio::io_context io_context(1);

        asio::signal_set signals(io_context, SIGINT, SIGTERM);
        signals.async_wait([&](auto, auto) {
            info("termination signal received, terminating...");
            io_context.stop();
        });

        // Create the acceptor to listen for incoming connections
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));

        // Enter the main loop
        co_spawn(io_context, listener(std::move(acceptor)), detached);
        co_spawn(io_context, print_bandwidth(), detached);

        io_context.run();
    } catch (std::exception const &e) { error("unresolved exception: {}", e.what()); }
}
