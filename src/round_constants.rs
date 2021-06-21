pub use crate::Error;
pub use bellperson::bls::Fr as Scalar;
use ff::{PrimeField, PrimeFieldDecodingError, PrimeFieldRepr, ScalarEngine};

/// From the paper ():
/// The round constants are generated using the Grain LFSR [23] in a self-shrinking
/// mode:
/// 1. Initialize the state with 80 bits b0, b1, . . . , b79, where
/// (a) b0, b1 describe the field,
/// (b) bi for 2 ≤ i ≤ 5 describe the S-Box,
/// (c) bi for 6 ≤ i ≤ 17 are the binary representation of n,
/// (d) bi for 18 ≤ i ≤ 29 are the binary representation of t,
/// (e) bi for 30 ≤ i ≤ 39 are the binary representation of RF ,
/// (f) bi for 40 ≤ i ≤ 49 are the binary representation of RP , and
/// (g) bi for 50 ≤ i ≤ 79 are set to 1.
/// 2. Update the bits using bi+80 = bi+62 ⊕ bi+51 ⊕ bi+38 ⊕ bi+23 ⊕ bi+13 ⊕ bi
/// .
/// 3. Discard the first 160 bits.
/// 4. Evaluate bits in pairs: If the first bit is a 1, output the second bit. If it is a
/// 0, discard the second bit.
/// Using this method, the generation of round constants depends on the specific
/// instance, and thus different round constants are used even if some of the chosen
/// parameters (e.g., n and t) are the same.
/// If a randomly sampled integer is not in Fp, we discard this value and take the
/// next one. Note that cryptographically strong randomness is not needed for the
/// round constants, and other methods can also be used.

/// Following https://extgit.iaik.tugraz.at/krypto/hadeshash/blob/master/code/scripts/create_rcs_grain.sage
pub fn generate_constants<E: ScalarEngine>(
    field: u8,
    sbox: u8,
    field_size: u16,
    t: u16,
    r_f: u16,
    r_p: u16,
) -> Vec<E::Fr> {
    let num_constants = (r_f + r_p) * t;
    let mut init_sequence: Vec<bool> = Vec::new();
    append_bits(&mut init_sequence, 2, field); // Bits 0-1
    append_bits(&mut init_sequence, 4, sbox); // Bits 2-5
    append_bits(&mut init_sequence, 12, field_size); // Bits 6-17
    append_bits(&mut init_sequence, 12, t); // Bits 18-29
    append_bits(&mut init_sequence, 10, r_f); // Bits 30-39
    append_bits(&mut init_sequence, 10, r_p); // Bits 40-49
    append_bits(&mut init_sequence, 30, 0b111111111111111111111111111111u128); // Bits 50-79

    let mut grain = Grain::new(init_sequence, field_size);
    let mut round_constants: Vec<E::Fr> = Vec::new();
    match field {
        1 => {
            let element_bytes = (field_size / 8) + ((field_size % 8) > 0) as u16;
            for _ in 0..num_constants {
                while {
                    // Smallest number of bytes which will hold one field element.
                    let mut bytes = vec![0u8; element_bytes as usize];
                    grain.get_next_bytes(&mut bytes);
                    if let Ok(f) = bytes_into_fr::<E>(&bytes) {
                        round_constants.push(f);
                        false
                    } else {
                        true
                    }
                } {}
            }
        }
        _ => {
            panic!("Only prime fields are supported.");
        }
    }
    round_constants
}

fn append_bits<T: Into<u128>>(vec: &mut Vec<bool>, n: usize, from: T) {
    let val = from.into() as u128;
    for i in (0..n).rev() {
        vec.push((val >> i) & 1 != 0);
    }
}

struct Grain {
    state: Vec<bool>,
    field_size: u16,
}

impl Grain {
    fn new(init_sequence: Vec<bool>, field_size: u16) -> Self {
        assert_eq!(80, init_sequence.len());
        let mut g = Grain {
            state: init_sequence,
            field_size,
        };
        for _ in 0..160 {
            g.generate_new_bit();
        }
        assert_eq!(80, g.state.len());
        g
    }

    fn generate_new_bit(&mut self) -> bool {
        let new_bit =
            self.bit(62) ^ self.bit(51) ^ self.bit(38) ^ self.bit(23) ^ self.bit(13) ^ self.bit(0);
        self.state.remove(0);
        self.state.push(new_bit);
        new_bit
    }

    fn bit(&self, index: usize) -> bool {
        self.state[index]
    }

    fn next_byte(&mut self, bit_count: usize) -> u8 {
        // Accumulate bits from most to least significant, so the most significant bit is the one generated first by the bit stream.
        let mut acc: u8 = 0;
        self.take(bit_count).for_each(|bit| {
            acc <<= 1;
            if bit {
                acc += 1;
            }
        });

        acc
    }

    fn get_next_bytes(&mut self, result: &mut [u8]) {
        let full_bytes = self.field_size as usize / 8;
        let remainder_bits = self.field_size as usize % 8;

        // Prime fields will always have remainder bits,
        // but other field types could be supported in the future.
        #[allow(clippy::needless_range_loop)]
        if remainder_bits > 0 {
            // If there is an unfull byte, it should be the first.
            result[0] = self.next_byte(remainder_bits);

            // Subsequent bytes are packed into result in the order generated.
            for i in 1..=full_bytes {
                result[i] = self.next_byte(8);
            }
        } else {
            for i in 0..full_bytes {
                result[i] = self.next_byte(8);
            }
        }
    }
}

impl Iterator for Grain {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let mut new_bit = self.generate_new_bit();
        while !new_bit {
            let _new_bit = self.generate_new_bit();
            new_bit = self.generate_new_bit();
        }
        new_bit = self.generate_new_bit();
        Some(new_bit)
    }
}

#[allow(dead_code)]
#[inline]
fn bool_to_u8(bit: bool, offset: usize) -> u8 {
    if bit {
        1u8 << offset
    } else {
        0u8
    }
}

/// Converts a slice of bools into their byte representation, in little endian.
#[allow(dead_code)]
pub fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|bits| {
            bool_to_u8(bits[7], 7)
                | bool_to_u8(bits[6], 6)
                | bool_to_u8(bits[5], 5)
                | bool_to_u8(bits[4], 4)
                | bool_to_u8(bits[3], 3)
                | bool_to_u8(bits[2], 2)
                | bool_to_u8(bits[1], 1)
                | bool_to_u8(bits[0], 0)
        })
        .collect()
}

// Takes a slice of bytes and returns an Fr if byte slice is exactly 32 bytes and does not overflow.
// Otherwise, returns a BadFrBytesError.
fn bytes_into_fr<E: ScalarEngine>(bytes: &[u8]) -> Result<E::Fr, PrimeFieldDecodingError> {
    assert_eq!(bytes.len(), 32);

    let mut fr_repr = <<<E as ScalarEngine>::Fr as PrimeField>::Repr as Default>::default();
    fr_repr
        // Try to read one field element from big-endian bytes.
        // Bytes are big-endian to agree with the integers generated by grain_random_bits in the reference implementation:
        //
        // def grain_random_bits(num_bits):
        //     random_bits = [grain_gen.next() for i in range(0, num_bits)]
        //     random_int = int("".join(str(i) for i in random_bits), 2)
        //     return random_int
        .read_be(bytes)
        .map_err(|e| PrimeFieldDecodingError::NotInField(e.to_string()))?;

    E::Fr::from_repr(fr_repr)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use bellperson::bls::Bls12;
    use serde_json::Value;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    #[test]
    fn test_round_constants() {
        // Bls12_381 modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
        // In hex: 73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
        let path = Path::new("parameters/round_constants-1-1-255-9-8-57-73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001.txt");
        let input = File::open(path).unwrap();
        let buffered = BufReader::new(input);
        let line = buffered.lines().skip(8).next().unwrap().unwrap();
        let replaced = line.replace("'", "\"");
        let parsed: Vec<Value> = serde_json::from_str(&replaced).unwrap();

        let expected = parsed.iter().map(|x| {
            if let Value::String(s) = x {
                s
            } else {
                panic!("Could not parse round constant string.")
            }
        });

        let generated = generate_constants::<Bls12>(1, 1, 255, 9, 8, 57)
            .iter()
            .map(|x| {
                let s = x.to_string();
                let start = s.find('(').unwrap() + 1;
                s[start..s.len() - 1].to_string()
            })
            .collect::<Vec<_>>();

        assert_eq!(expected.len(), generated.len());

        generated
            .iter()
            .zip(expected)
            .for_each(|(generated, expected)| assert_eq!(generated, expected));
    }
}
