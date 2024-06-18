use std::io;
use std::net::IpAddr::{V4, V6};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use env_logger::Env;
use hickory_resolver::config::*;
use hickory_resolver::TokioAsyncResolver;
use log::{error, info};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time;

#[derive(Parser, Debug)]
#[clap(name = "mimp-rs", version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value_t = 23333)]
    port: u16,
}

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        // Setup the logger from the environment
        env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

        let args = Args::parse();
        if let Err(e) = execute(args).await {
            error!("{:}", e);
            return Err(e);
        }

        Ok(())
    })
}

async fn execute(args: Args) -> Result<()> {
    let Args { port, .. } = args;
    info!("starting mimp-rs on port {port}...");

    // Bind the listener to the port
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .with_context(|| format!("failed to bind to port {port}"))?;

    // Create the DNS resolver
    let resolver = TokioAsyncResolver::tokio(ResolverConfig::default(), ResolverOpts::default());
    let resolver = Arc::new(resolver);

    loop {
        let (stream, _) = listener.accept().await?;
        let resolver = resolver.clone();

        tokio::spawn(async move {
            if let Err(e) = process(stream, resolver).await {
                error!("{}", e);
            }
        });
    }
}

/// Reads a fixed number of bytes from a TCP stream with a timeout.
///
/// # Arguments
///
/// * `stream` - The TCP stream to read from.
/// * `buffer` - The buffer to store the read bytes.
/// * `timeout` - The timeout duration.
///
/// # Returns
///
/// Returns a `Result` indicating success or failure.
async fn timed_read_exact(
    stream: &mut TcpStream,
    buffer: &mut [u8],
    timeout: time::Duration,
) -> Result<()> {
    match time::timeout(timeout, async {
        stream.read_exact(buffer).await?;
        io::Result::Ok(())
    })
    .await
    {
        Ok(Err(e)) => Err(anyhow::Error::from(e)),
        Err(_) => {
            bail!("timed out after {timeout:?} reading from stream");
        }
        _ => Ok(()),
    }
}

/// Write a fixed number of bytes to a TCP stream with a timeout.
///
/// # Arguments
///
/// * `stream` - The TCP stream to write to.
/// * `buffer` - The buffer to store the read bytes.
/// * `timeout` - The timeout duration.
///
/// # Returns
///
/// Returns a `Result` indicating success or failure.
async fn timed_write_and_flush(
    stream: &mut TcpStream,
    buffer: &[u8],
    timeout: time::Duration,
) -> Result<()> {
    match time::timeout(timeout, async {
        stream.write(buffer).await?;
        stream.flush().await?;
        io::Result::Ok(())
    })
    .await
    {
        Ok(Err(e)) => Err(anyhow::Error::from(e)),
        Err(_) => {
            bail!("timed out after {timeout:?} writing to stream");
        }
        _ => Ok(()),
    }
}

async fn process(mut stream: TcpStream, resolver: Arc<TokioAsyncResolver>) -> Result<()> {
    info!(
        "new connection from {addr}",
        addr = stream.peer_addr().unwrap()
    );

    let timeout = time::Duration::from_secs(2);
    let mut buffer = [0; 4096];

    /////////////////////////////////////////////////////////////////////////////
    // The client connects to the server, and sends a version identifier/method
    // selection message:
    //           +----+----------+----------+
    //           |VER | NMETHODS | METHODS  |
    //           +----+----------+----------+
    //           | 1  |    1     | 1 to 255 |
    //           +----+----------+----------+
    timed_read_exact(&mut stream, &mut buffer[0..2], timeout).await?;
    if buffer[0] != 0x05 {
        bail!("invalid SOCKS version: {version}", version = buffer[0]);
    }

    let n_methods = buffer[1];
    timed_read_exact(&mut stream, &mut buffer[0..(n_methods as usize)], timeout).await?;

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
    // TODO(zike): implement this part
    timed_write_and_flush(&mut stream, &[0x05, 0x00], timeout).await?;

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
    timed_read_exact(&mut stream, &mut buffer[0..4], timeout).await?;

    if buffer[0] != 0x05 {
        bail!("invalid SOCKS version: {version}", version = buffer[0]);
    }

    if buffer[1] != 0x01 {
        bail!("unsupported command: {command}", command = buffer[1]);
    }

    if buffer[2] != 0x00 {
        bail!("invalid reserved field: {reserved}", reserved = buffer[2]);
    }

    let atyp = buffer[3];
    let (addr, port, human_readable) = match atyp {
        0x01 => {
            timed_read_exact(&mut stream, &mut buffer[0..6], timeout).await?;
            let addr = Ipv4Addr::new(buffer[0], buffer[1], buffer[2], buffer[3]);
            let port = u16::from_be_bytes([buffer[4], buffer[5]]);

            (V4(addr), port, addr.to_string())
        }
        0x03 => {
            timed_read_exact(&mut stream, &mut buffer[0..1], timeout).await?;
            let domain_len = buffer[0] as usize;

            timed_read_exact(
                &mut stream,
                &mut buffer[0..(domain_len as usize + 2)],
                timeout,
            )
            .await?;

            let domain = String::from_utf8_lossy(&buffer[0..domain_len]).into_owned();
            let port = u16::from_be_bytes([buffer[domain_len], buffer[domain_len + 1]]);

            // Perform bounded DNS resolution
            info!("resolving {domain}...");
            let start = Instant::now();
            let ip = match time::timeout(time::Duration::from_secs(5), async {
                // This returns a `Result<IpAddr, anyhow::Error>`
                match resolver.lookup_ip(&domain).await?.into_iter().next() {
                    Some(ip) => Ok(ip),
                    None => Err(anyhow!("failed to resolve domain: {domain}")),
                }
            })
            .await
            {
                Ok(x) => x?,
                Err(_) => bail!("failed to resolve domain {domain} after 5 seconds"),
            };
            info!(
                "{} ==(resolves to)==> {} in {} ms",
                domain,
                ip,
                start.elapsed().as_millis()
            );
            (ip, port, format!("{domain}:{port}"))
        }
        0x04 => {
            let addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0);
            timed_read_exact(&mut stream, &mut addr.octets(), timeout).await?;
            timed_read_exact(&mut stream, &mut buffer[0..2], timeout).await?;

            (
                V6(addr),
                u16::from_be_bytes([buffer[0], buffer[1]]),
                addr.to_string(),
            )
        }
        _ => bail!(
            "unsupported address type (ATYP): {address_type}",
            address_type = buffer[3]
        ),
    };

    info!("connecting to {human_readable}...");

    let mut target = match time::timeout(time::Duration::from_secs(5), async {
        TcpStream::connect((addr, port)).await
    })
    .await
    {
        Ok(x) => x?,
        Err(_) => bail!("failed to connect {human_readable} after 5 seconds"),
    };

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
    timed_write_and_flush(
        &mut stream,
        &[0x05, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        timeout,
    )
    .await?;

    info!("relay established for inbound <=> {human_readable}");
    tokio::io::copy_bidirectional(&mut stream, &mut target).await?;
    Ok(())
}
