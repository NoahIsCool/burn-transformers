//! Compare semantic masks and absolute positions to small scalar reference formulas.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use transformers::{
    TokenMask, attention::padding_attention_mask, position::derive_batched_position_ids,
};

/// Constructs the backend's native Boolean representation from small integer indicators.
fn mask(values: &[bool], batch: usize, sequence: usize) -> TokenMask<Flex> {
    let values = values
        .iter()
        .map(|&value| i32::from(value))
        .collect::<Vec<_>>();
    TokenMask::new(
        Tensor::<Flex, 2, Int>::from_data(
            TensorData::new(values, [batch, sequence]),
            &Default::default(),
        )
        .bool(),
    )
}

/// Checks mask polarity, broadcast axes, malformed shapes, padding offsets and integer overflow.
pub fn exercise(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    let batch = usize::from(bytes.byte() % 4) + 1;
    let queries = usize::from(bytes.byte() % 16) + 1;
    let keys = usize::from(bytes.byte() % 16) + 1;
    let heads = usize::from(bytes.byte() % 4) + 1;
    let start = (0..batch).map(|_| bytes.size()).collect::<Vec<_>>();
    let masked_position = bytes.size();
    let query = (0..batch * queries)
        .map(|_| bytes.byte() & 1 == 1)
        .collect::<Vec<_>>();
    let key = (0..batch * keys)
        .map(|_| bytes.byte() & 1 == 1)
        .collect::<Vec<_>>();
    let query_mask = mask(&query, batch, queries);
    let key_mask = mask(&key, batch, keys);
    for (use_query, use_key) in [(true, true), (true, false), (false, true), (false, false)] {
        let result = padding_attention_mask(
            use_query.then_some(&query_mask),
            use_key.then_some(&key_mask),
            batch,
            heads,
            queries,
            keys,
        )
        .expect("bounded matching mask geometry");
        if !use_query && !use_key {
            assert!(result.is_none());
            continue;
        }
        let values = result.unwrap().int().into_data().to_vec::<i32>().unwrap();
        let mut index = 0;
        for row in 0..batch {
            for _head in 0..heads {
                for q in 0..queries {
                    for k in 0..keys {
                        let excluded = (use_query && !query[row * queries + q])
                            || (use_key && !key[row * keys + k]);
                        assert_eq!(values[index] != 0, excluded);
                        index += 1;
                    }
                }
            }
        }
    }
    assert!(
        padding_attention_mask(
            Some(&query_mask),
            Some(&key_mask),
            batch + 1,
            heads,
            queries,
            keys
        )
        .is_err()
    );
    assert!(
        padding_attention_mask(Some(&query_mask), Some(&key_mask), batch, 0, queries, keys)
            .is_err()
    );

    let positions = derive_batched_position_ids(&query, batch, queries, &start, masked_position);
    let mut expected = Vec::with_capacity(query.len());
    let mut overflow = false;
    for row in 0..batch {
        let mut ordinal = 0_u128;
        for &valid in &query[row * queries..(row + 1) * queries] {
            let position = if valid {
                let position = start[row] as u128 + ordinal;
                ordinal += 1;
                position
            } else {
                masked_position as u128
            };
            overflow |= position > usize::MAX as u128;
            expected.push(position as usize);
        }
    }
    if overflow {
        assert!(positions.is_err());
    } else {
        assert_eq!(positions.unwrap(), expected);
    }
    assert!(
        derive_batched_position_ids(&query[..query.len() - 1], batch, queries, &start, 0).is_err()
    );
    assert!(derive_batched_position_ids(&query, batch, queries, &start[..batch - 1], 0).is_err());
    assert!(derive_batched_position_ids(&[], usize::MAX, 2, &[], 0).is_err());
}

#[cfg(test)]
#[path = "masks_tests.rs"]
mod tests;
