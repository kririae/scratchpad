#define FMT_HEADER_ONLY
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#define SPDLOG_FMT_EXTERNAL
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <argparse/argparse.hpp>
#include <asio.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <chrono>
#include <map>
#include <regex>
#include <source_location>

using spdlog::debug;
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
/// The number of active connections
std::atomic_uint64_t active_conn{0};

/// Timeout that applies to the relay, *can* be modified
steady_clock::duration relay_timeout{};

/// Timeout that applies to async_connect, should be modified
steady_clock::duration connect_timeout{};

/// Timeout that applies to general async_read/async_write, should be modified
steady_clock::duration general_timeout{};

// The pair of valid users
std::map<std::string, std::string> auth_table;

//
//
// The transferred bytes in the last cycle
std::atomic_uint64_t up_bytes{0};
std::atomic_uint64_t down_bytes{0};
bool is_last_non_zero{true};

//
//
struct BridgeConfig {
    std::string user;
    std::string passwd;

    std::string host;
    int port;
};

//
std::atomic_uint32_t bridge_cnt{0};
std::vector<BridgeConfig> bridges;

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

template <typename AsyncReadStream, typename MutableBufferSequence, typename WriteToken>
awaitable<void> timed_async_read(
    AsyncReadStream &s, MutableBufferSequence const &buffers, WriteToken &&token,
    std::source_location loc = std::source_location::current()
) {
    co_await stop_after(
        general_timeout, asio::async_read(s, buffers, std::forward<WriteToken>(token)),
        [&] {
        error("async_read is timeout-ed after {} s at line {}", general_timeout / 1.0s, loc.line());
        throw std::runtime_error("async_read is timeout-ed");
    }
    );
}

template <typename AsyncWriteStream, typename ConstBufferSequence, typename WriteToken>
awaitable<void> timed_async_write(
    AsyncWriteStream &s, ConstBufferSequence const &buffers, WriteToken &&token,
    std::source_location loc = std::source_location::current()
) {
    co_await stop_after(
        general_timeout, asio::async_write(s, buffers, std::forward<WriteToken>(token)),
        [&] {
        error(
            "async_write is timeout-ed after {} s at line {}", general_timeout / 1.0s, loc.line()
        );
        throw std::runtime_error("async_read is timeout-ed");
    }
    );
}

/// \brief Validate the authentication of the socks5 request.
awaitable<bool> validate_auth(tcp::socket &client) noexcept try {
    // It begins with the client producing a Username/Password request:
    //         +----+------+----------+------+----------+
    //         |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
    //         +----+------+----------+------+----------+
    //         | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
    //         +----+------+----------+------+----------+
    uint8_t ver{0};
    uint8_t ulen{0};
    co_await timed_async_read(client, buffer(&ver, 1), use_awaitable);
    co_await timed_async_read(client, buffer(&ulen, 1), use_awaitable);

    std::string uname(ulen, '\0');
    co_await timed_async_read(client, buffer(uname), use_awaitable);

    uint8_t plen{0};
    co_await timed_async_read(client, buffer(&plen, 1), use_awaitable);

    std::string passwd(plen, '\0');
    co_await timed_async_read(client, buffer(passwd), use_awaitable);

    // Check the authentication table
    if (auth_table.contains(uname) && auth_table[uname] == passwd)
        co_return true;
    else
        co_return false;
} catch (std::exception const &e) {
    error("unresolved socks5 auth exception: {}", e.what());
    co_return false;
}

/// \brief Send the authentication to the socks5 server asynchronously
awaitable<bool> send_auth(tcp::socket &server, BridgeConfig const &bridge, auto &buf) noexcept try {
    // Construct the Username/Password request
    //         +----+------+----------+------+----------+
    //         |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
    //         +----+------+----------+------+----------+
    //         | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
    //         +----+------+----------+------+----------+
    assert(!bridge.user.empty());
    assert(!bridge.passwd.empty());
    if (bridge.user.size() > 255 || bridge.passwd.size() > 255) {
        error("username or password is too long");
        co_return true;
    }

    buf[0] = 0x05;
    buf[1] = bridge.user.size();
    std::copy(bridge.user.begin(), bridge.user.end(), buf.begin() + 2);

    size_t const plen_offset = 2 + bridge.user.size();
    buf[plen_offset] = bridge.passwd.size();
    std::copy(bridge.passwd.begin(), bridge.passwd.end(), buf.begin() + plen_offset + 1);

    size_t const total_len = 3 + bridge.user.size() + bridge.passwd.size();
    assert(total_len <= 1024);

    co_await timed_async_write(server, buffer(buf, total_len), use_awaitable);

    // Receive the response
    co_await timed_async_read(server, buffer(buf, 2), use_awaitable);

    if (buf[0] != 0x05) {
        error("invalid socks5 version from server: {}", buf[0]);
        co_return true;
    }

    if (buf[1] != 0x00) {
        error("failed to authenticate the bridge '{}:{}'", buf[1], bridge.host, bridge.port);
        co_return true;
    }

    co_return false;
} catch (std::exception const &e) {
    error("unresolved socks5 auth exception: {}", e.what());
    co_return true;
}

/// \brief Respond to the socks5 authentication asynchronously
awaitable<void> reply_auth(uint8_t reply_type, tcp::socket &client) {
    co_await timed_async_write(client, buffer<uint8_t>({0x05, reply_type}), use_awaitable);
}

/// \brief Parse the ATYP field of the socks5 request
awaitable<std::optional<tcp::resolver::query>>
parse_atyp(uint8_t atyp_type, tcp::socket &client) noexcept try {
    // Parse the ATYP field
    // In an address field (DST.ADDR, BND.ADDR), the ATYP field specifies the
    // type of address contained within the field as follows:
    switch (atyp_type) {
    default: error("invalid socks5 address type: {}", atyp_type); co_return std::nullopt;
    case 0x01: {
        // the address is a version-4 IP address, with a length of 4 octets
        uint16_t port{0};
        asio::detail::array<uint8_t, 4> ipv4_address;
        co_await timed_async_read(client, buffer(ipv4_address), use_awaitable);
        co_await timed_async_read(client, buffer(&port, 2), use_awaitable);

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
        co_await timed_async_read(client, buffer(&domain_length, 1), use_awaitable);

        uint16_t port{0};
        std::string domain_name(domain_length, '\0');
        co_await timed_async_read(client, buffer(domain_name), use_awaitable);
        co_await timed_async_read(client, buffer(&port, 2), use_awaitable);

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
        co_await timed_async_read(client, buffer(ipv6_address), use_awaitable);
        co_await timed_async_read(client, buffer(&port, 2), use_awaitable);

        port = ntohs(port);
        co_return tcp::resolver::query{
            /*host=    */ asio::ip::make_address_v6(ipv6_address).to_string(),
            /*service= */ std::to_string(port)
        };
        break;
    }
    }

    co_return std::nullopt;
} catch (std::exception &e) {
    // do nothing
    error("unresolved ATYP parsing exception: {}", e.what());
    co_return std::nullopt;
}

/// \brief Respond to the socks5 request asynchronously
awaitable<void> reply_conn_req(uint8_t reply_type, tcp::socket &client) {
    co_await timed_async_write(
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
    tcp::socket &to, tcp::socket &from, steady_clock::time_point &deadline, auto &buf,
    std::function<void(size_t)> callback = nullptr
) noexcept {
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
awaitable<void> copy_bidirectional(tcp::socket &client, tcp::socket &server, auto &buf) noexcept {
    steady_clock::time_point client_to_server_deadline{steady_clock::now() + relay_timeout};
    steady_clock::time_point server_to_client_deadline{steady_clock::now() + relay_timeout};

    auto update_up_bytes = [&](size_t n1) { up_bytes.fetch_add(n1); };
    auto update_down_bytes = [&](size_t n1) { down_bytes.fetch_add(n1); };

    // If timeout is set to zero, disable watchdog
    if (relay_timeout == 0s) {
        co_await (
            copy_directional(client, server, server_to_client_deadline, buf, update_down_bytes) &&
            copy_directional(server, client, client_to_server_deadline, buf, update_up_bytes)
        );
    } else {
        co_await (
            (copy_directional(client, server, server_to_client_deadline, buf, update_down_bytes) ||
             watchdog(
                 server_to_client_deadline,
                 [] { warn("relay is timeout-ed after {} s", relay_timeout / 1.0s); }
             )) &&
            (copy_directional(server, client, client_to_server_deadline, buf, update_up_bytes) ||
             watchdog(client_to_server_deadline))
        );
    }
}

/// \brief Connect to the remote server asynchronously
awaitable<std::optional<tcp::socket>>
connect_server(tcp::socket &client, tcp::resolver::query query, auto &buf) {
    tcp::socket server(co_await asio::this_coro::executor);
    tcp::resolver resolver(co_await asio::this_coro::executor);
    auto const op = co_await stop_after(
        connect_timeout,
        asio::async_connect(server, resolver.resolve(query), use_nothrow_awaitable),
        [&] {
        warn(
            "connecting to '{}:{}' timeout after {}", query.host_name(), query.service_name(),
            connect_timeout
        );
    }
    );

    if (!op.has_value()) {
        co_await reply_conn_req(0x04 /* Host unreachable */, client);
        co_return std::nullopt;
    }

    auto [e, endpoint] = op.value();
    if (e) {
        warn(
            "failed to get connected to the server at '{}:{}'", query.host_name(),
            query.service_name()
        );
        co_await reply_conn_req(0x05 /* connection refused */, client);
        co_return std::nullopt;
    }

    co_return std::move(server);
}

/// \brief Receive and parse the socks5 request asynchronously
awaitable<void> handle_socks5(tcp::socket client) try {
    active_conn.fetch_add(1);
    struct Finally {
        ~Finally() { active_conn.fetch_sub(1); }
    } finally;

    std::array<uint8_t, 1024> buf;
    co_await timed_async_read(client, buffer(buf, 2), use_awaitable);

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
    co_await timed_async_read(client, buffer(buf, nmethods), use_awaitable);

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
    auto const auth_required = !auth_table.empty();

    auto c_noauth_supported = false;
    auto c_auth_supported = false;
    for (uint8_t i = 0; i < nmethods; ++i)
        if (buf[i] == 0x00)
            c_noauth_supported = true;
        else if (buf[i] == 0x02)
            c_auth_supported = true;

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
        if (!c_auth_supported) {
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
                "authentication failed at '{}:{}'", client.remote_endpoint().address().to_string(),
                client.remote_endpoint().port()
            );
            co_return;
        }
    } else /* auth not required */
        if (c_noauth_supported) {
            // authentication is not required, and no authentication is supported
            co_await reply_method_selection(0x00, client);
        } else if (c_auth_supported) {
            // Deal with the weird case that auth is not required, but only auth is
            // supported.
            bool const should_terminate = co_await auth_and_reply();
            if (should_terminate) {
                warn(
                    "authentication failed at '{}:{}'",
                    client.remote_endpoint().address().to_string(), client.remote_endpoint().port()
                );
                co_return;
            }
        } else {
            warn("no acceptable socks5 method");
            co_await reply_method_selection(0xFF, client);
            co_return;
        }

    if (bridges.empty()) {
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
        co_await timed_async_read(client, buffer(buf, 4), use_awaitable);
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
        auto server = co_await connect_server(client, query.value(), buf);
        if (!server.has_value())
            co_return;
        // Respond to the client that the connection is established
        co_await reply_conn_req(0x00 /* succeeded */, client);

        // Actually execute the bi-directional copy
        info("relay to '{}:{}' is established", query->host_name(), query->service_name());

        co_await copy_bidirectional(client, server.value(), buf);

        info(
            "relay to '{}:{}' is closed, lasting {} connections active", query->host_name(),
            query->service_name(), active_conn.load() - 1
        );
    } else {
        // Enter bridge mode
        auto const bridge_idx = bridge_cnt.fetch_add(1) % bridges.size();
        auto const bridge = bridges[bridge_idx];

        auto server = co_await connect_server(
            client, tcp::resolver::query{bridge.host, std::to_string(bridge.port)}, buf
        );
        if (!server.has_value())
            // Responses are handled in the `connect_server` function
            co_return;

        //  o  X'00' NO AUTHENTICATION REQUIRED
        //  o  X'01' GSSAPI
        //  o  X'02' USERNAME/PASSWORD
        //  o  X'03' to X'7F' IANA ASSIGNED
        //  o  X'80' to X'FE' RESERVED FOR PRIVATE METHODS
        if (bridge.user.empty()) {
            co_await timed_async_write(
                server.value(), buffer<uint8_t>({0x05, 0x01, 0x00}), use_awaitable
            );
        } else {
            co_await timed_async_write(
                server.value(), buffer<uint8_t>({0x05, 0x02, 0x00, 0x02}), use_awaitable
            );
        }

        co_await timed_async_read(server.value(), buffer(buf, 2), use_awaitable);
        if (buf[0] != 0x05) {
            error("invalid socks5 version from server: {}", buf[0]);
            co_return;
        }

        // The server requires authentication
        if (buf[1] == 0x02) {
            if (bridge.user.empty()) {
                error("server requires authentication, but no user is provided");
                co_return;
            } else {
                bool const should_terminate = co_await send_auth(server.value(), bridge, buf);
                if (should_terminate)
                    co_return;
            }
        } else if (buf[1] != 0x00) {
            error("invalid socks5 authentication method from server: {}", buf[1]);
            co_return;
        }

        info("bridge to '{}:{}' is established", bridge.host, bridge.port);
        co_await copy_bidirectional(client, server.value(), buf);
        info(
            "bridge to '{}:{}' is closed, lasting {} connections active", bridge.host, bridge.port,
            active_conn.load() - 1
        );
    }
} catch (asio::system_error const &e) {
    // Ignore EOF exception
    if (e.code() != asio::error::eof)
        warn("unresolved socks5 handler asio exception: {}", e.code().message());
} catch (std::exception const &e) {
    warn("unresolved socks5 handler local exception: {}", e.what());
}

awaitable<void> dispatch_connection(tcp::socket request) {
    co_await handle_socks5(std::move(request));
}

awaitable<void> listener(tcp::acceptor acceptor) {
    auto local_endpoint = acceptor.local_endpoint();
    info("listening on '{}:{}'", local_endpoint.address().to_string(), local_endpoint.port());
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

int main(int argc, char *argv[]) try {
    spdlog::cfg::load_env_levels();
    argparse::ArgumentParser program("mimp", "0.1.1");

    program.add_argument("-p", "--port")
        .help("the port to listen to (23333 by default)")
        .default_value(23333)
        .scan<'i', int>();

    program.add_argument("-a", "--auth")
        .help("add users to enable authentication in the format of `user:passwd`")
        .nargs(1)
        .action([&](std::string const &value) {
        std::regex auth_regex("^([^:]+):(.+)$");
        std::smatch match;
        if (std::regex_match(value, match, auth_regex) && match.size() == 3)
            auth_table[match[1].str()] = match[2].str();
        else
            throw std::runtime_error(
                "invalid auth specification, should be in the format of `user:passwd`"
            );
    }).append();

    program.add_argument("-b", "--bridge")
        .help("enable bridge mode and specify the target socks5 server to connect in the format of "
              "[user:passwd@]host:port")
        .nargs(1)
        .action([&](std::string const &value) {
        std::regex bridge_regex("^(?:([^:]+):([^@]+)@)?([^:]+):([0-9]+)$");
        std::smatch match;
        if (std::regex_match(value, match, bridge_regex) &&
            (match.size() == 5 || match.size() == 3)) {
            BridgeConfig config;
            if (match.size() == 5) {
                config.user = match[1].str();
                config.passwd = match[2].str();
                config.host = match[3].str();
                config.port = std::stoi(match[4].str());
            } else {
                config.host = match[1].str();
                config.port = std::stoi(match[2].str());
            }

            bridges.push_back(config);
        } else
            throw std::runtime_error(
                "invalid bridge specification, should be in the format of `[user:passwd@]host:port`"
            );
    }).append();

    program.add_argument("-j", "--jobs")
        .help("the number of jobs to run concurrently (1 by default)")
        .default_value(1)
        .scan<'i', int>();

    program.add_argument("--relay-timeout")
        .help("the timeout delay of a relay in ms (10000ms by default)")
        .default_value(10000)
        .scan<'i', int>();

    program.add_argument("--connect-timeout")
        .help("the timeout delay of connect request in ms (5000ms by default)")
        .default_value(5000)
        .scan<'i', int>();

    program.add_argument("--general-timeout")
        .help("the timeout delay of a general request in ms (5000ms by default)")
        .default_value(5000)
        .scan<'i', int>();

    program.parse_args(argc, argv);

    if (program["--help"] == true) {
        std::cout << program;
        return 0;
    }

    auto port = program.get<int>("--port");
    if (port < 0 || port > 65535)
        throw std::out_of_range("invalid port specification");

    auto jobs = program.get<int>("--jobs");
    if (jobs <= 0)
        throw std::out_of_range("invalid job specification");

    auto t_relay_timeout = program.get<int>("--relay-timeout");
    if (t_relay_timeout < 0)
        throw std::out_of_range("invalid timeout specification");
    relay_timeout = std::chrono::milliseconds(t_relay_timeout);

    auto t_connect_timeout = program.get<int>("--connect-timeout");
    if (t_connect_timeout < 0)
        throw std::out_of_range("invalid timeout specification");
    connect_timeout = std::chrono::milliseconds(t_connect_timeout);

    auto t_general_timeout = program.get<int>("--general-timeout");
    if (t_general_timeout < 0)
        throw std::out_of_range("invalid timeout specification");
    general_timeout = std::chrono::milliseconds(t_general_timeout);

    auto print_timeout = [](auto timeout, std::string_view name) {
        if (timeout == 0s)
            info("{:s} timeout is disabled", name);
        else
            info("{:s} timeout is set to be {} s", name, timeout / 1.0s);
    };

    print_timeout(relay_timeout, "relay");
    print_timeout(connect_timeout, "connect");
    print_timeout(general_timeout, "general");

    if (!bridges.empty()) {
        info("bridge mode is enabled");
        int bridge_cnt = 0;
        for (auto &bridge : bridges) {
            // if there's no user specified
            if (bridge.user.empty())
                debug(
                    "bridge target {:d}: host: {:s}, port: {:d}", bridge_cnt, bridge.host,
                    bridge.port
                );
            else
                debug(
                    "bridge target {:d}: user: {:s}, passwd: {:s}, host: {:s}, port: {:d}",
                    bridge_cnt, bridge.user, bridge.passwd, bridge.host, bridge.port
                );
            ++bridge_cnt;
        }
    }

    if (auth_table.empty())
        info("authentication is not required");
    else {
        info("authentication is required");
        for (auto const &[user, passwd] : auth_table)
            debug("auth item: user: {:s}, passwd: {:s}", user, passwd);
    }

    // Create the I/O context that will run the coroutine
    asio::io_context io_context(jobs);

    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto) {
        info(
            "termination signal received with {} active connections, terminating...",
            active_conn.load()
        );
        io_context.stop();
    });

    // Create the acceptor to listen for incoming connections
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));

    // Enter the main loop
    co_spawn(io_context, listener(std::move(acceptor)), detached);
    co_spawn(io_context, print_bandwidth(), detached);

    io_context.run();
    return 0;
} catch (std::exception const &e) {
    error("unresolved exception: {}", e.what());
    return 1;
}
