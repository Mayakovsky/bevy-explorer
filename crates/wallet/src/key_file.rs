//! Emanate: log in as a real (non-guest) account from a private key held in a file.
//!
//! The key never touches the command line or the logs: the caller passes a *path*, this reads
//! the hex key from it, and builds the standard Decentraland identity: a fresh ephemeral key
//! whose `ECDSA_EPHEMERAL` link is signed (EIP-191 personal sign) by the root key. Nothing
//! here prints the key. Intended for a throwaway test key (Zone) or the Synth's own account.

use std::{path::Path, str::FromStr};

use alloy_signer::Signer;
use alloy_signer_local::PrivateKeySigner;
use anyhow::{anyhow, Context};
use common::structs::ChainLink;

use crate::Wallet;

/// How long the ephemeral delegation stays valid. A spike run is hours, not days; the
/// production Synth should refresh, so keep this bounded rather than "forever".
const EPHEMERAL_VALIDITY_DAYS: i64 = 30;

/// Reads a hex private key (optional `0x`, surrounding whitespace ignored) from `path`.
pub fn read_key_file(path: &Path) -> Result<PrivateKeySigner, anyhow::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .with_context(|| format!("wallet key file {}", path.display()))?
            .permissions()
            .mode();
        if mode & 0o077 != 0 {
            anyhow::bail!(
                "wallet key file {} is accessible to group/others (mode {:o}); chmod 600 it",
                path.display(),
                mode & 0o777
            );
        }
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading wallet key file {}", path.display()))?;
    PrivateKeySigner::from_str(raw.trim().trim_start_matches("0x"))
        .map_err(|_| anyhow!("wallet key file {} does not hold a valid hex private key", path.display()))
}

/// Builds the ephemeral delegation for `root`: (ephemeral signer, the one `ECDSA_EPHEMERAL` link).
pub async fn make_ephemeral_delegation(
    root: &PrivateKeySigner,
) -> Result<(PrivateKeySigner, ChainLink), anyhow::Error> {
    let ephemeral = PrivateKeySigner::random_with(&mut rand::thread_rng());
    let expiration =
        (chrono::Utc::now() + chrono::Duration::days(EPHEMERAL_VALIDITY_DAYS))
            .format("%Y-%m-%dT%H:%M:%S%.3fZ");
    let payload = format!(
        "Decentraland Login\nEphemeral address: {}\nExpiration: {}",
        ephemeral.address(),
        expiration
    );
    let signature = root
        .sign_message(payload.as_bytes())
        .await
        .map_err(|e| anyhow!("signing ephemeral delegation: {e}"))?;
    Ok((
        ephemeral,
        ChainLink {
            ty: "ECDSA_EPHEMERAL".to_owned(),
            payload,
            signature: signature.to_string(),
        },
    ))
}

/// Finalizes `wallet` as a real account from the key at `path`. Returns the root address.
pub fn finalize_from_key_file(
    wallet: &mut Wallet,
    path: &Path,
) -> Result<alloy_core::primitives::Address, anyhow::Error> {
    let root = read_key_file(path)?;
    let root_address = root.address();
    let (ephemeral, link) = futures_lite_block_on(make_ephemeral_delegation(&root))?;
    wallet.finalize(root_address, ephemeral, vec![link], false);
    Ok(root_address)
}

fn futures_lite_block_on<T>(
    fut: impl std::future::Future<Output = Result<T, anyhow::Error>>,
) -> Result<T, anyhow::Error> {
    async_std::task::block_on(fut)
}
