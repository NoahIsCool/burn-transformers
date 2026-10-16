//! Qwen3.5 MTP inventory policy.

use super::*;

/// Every request and layer count maps to exactly the former contract behavior.
#[test]
fn mtp_inventory_matches_the_former_policy() {
    for include_mtp in [false, true] {
        for dedicated_embeddings in [false, true] {
            for layers in [0, 1, 3] {
                let expected = if include_mtp && dedicated_embeddings {
                    Err(DedicatedMtpEmbeddings)
                } else if include_mtp {
                    Ok(if layers == 0 {
                        MtpInventory::Absent
                    } else {
                        MtpInventory::Load
                    })
                } else if layers > 0 {
                    Ok(MtpInventory::Exempt)
                } else {
                    Ok(MtpInventory::Absent)
                };
                assert_eq!(
                    mtp_inventory(include_mtp, dedicated_embeddings, layers),
                    expected
                );
            }
        }
    }
}
