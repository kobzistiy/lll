// main.rs

use rug::{Integer, Rational, ops::Pow};
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Путь к CSV-файлу с базисом
    #[arg(long)]
    file: Option<String>,

    /// Запустить тестовый режим (встроенный пример)
    #[arg(long)]
    test: bool,
    
    /// Данные в виде [["11","3","4"],["2","11","5"]...]
    #[arg(long)]
    data: Option<String>,
}

// --- Utility Functions for Integer Vectors ---

fn dot_product(v1: &[Integer], v2: &[Integer]) -> Integer {
    v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
}

fn subtract_vec(v1: &[Integer], v2: &[Integer]) -> Vec<Integer> {
    v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
}

fn scalar_mul(scalar: &Integer, v: &[Integer]) -> Vec<Integer> {
    v.iter().map(|x| scalar * x).collect()
}

// --- LLL Algorithm (Integer-based implementation) ---

fn lll(b: &mut Vec<Vec<Integer>>, delta: &Rational) {
    let n = b.len();
    let mut b_star = Vec::with_capacity(n); // Gram-Schmidt basis (b*)
    let mut mu = vec![vec![Rational::new(); n]; n]; // mu coefficients

    // Initial Gram-Schmidt
    for i in 0..n {
        let mut b_i_rational: Vec<Rational> = b[i].iter().map(Rational::from).collect();
        for j in 0..i {
            let num = b[i].iter().zip(b_star[j].iter()).map(|(bi, bs)| Rational::from(bi) * bs).sum::<Rational>();
            let den = b_star[j].iter().map(|c| c.clone().pow(2)).sum::<Rational>();
            mu[i][j] = num / den;
            let mu_b_star: Vec<Rational> = b_star[j].iter().map(|c| mu[i][j].clone() * c).collect();
            b_i_rational = b_i_rational.iter().zip(mu_b_star.iter()).map(|(a, b)| a - b).collect();
        }
        b_star.push(b_i_rational);
    }
    
    let mut k = 1;
    while k < n {
        // Size reduction
        for j in (0..k).rev() {
            let mu_kj = &mu[k][j];
            if mu_kj.abs() > 0.5 {
                let q = mu_kj.round(); // Rounds to nearest integer
                let q_integer = q.to_integer().unwrap();
                b[k] = subtract_vec(&b[k], &scalar_mul(&q_integer, &b[j]));

                // Update mu values for row k
                for i in 0..=j {
                   mu[k][i] -= q.clone() * mu[j][i].clone();
                }
            }
        }

        // Lovász condition
        let norm_b_star_k_sq: Rational = b_star[k].iter().map(|c| c.clone().pow(2)).sum();
        let norm_b_star_k_minus_1_sq: Rational = b_star[k-1].iter().map(|c| c.clone().pow(2)).sum();
        
        if norm_b_star_k_sq >= (delta - mu[k][k-1].clone().pow(2)) * norm_b_star_k_minus_1_sq {
            k += 1;
        } else {
            // Swap b_k and b_{k-1}
            b.swap(k, k-1);

            // Recompute Gram-Schmidt for the swapped vectors
            // (A full recomputation is easier to implement correctly)
            for i in 0..n {
                let mut b_i_rational: Vec<Rational> = b[i].iter().map(Rational::from).collect();
                for j in 0..i {
                    let num = b[i].iter().zip(b_star[j].iter()).map(|(bi, bs)| Rational::from(bi) * bs).sum::<Rational>();
                    let den = b_star[j].iter().map(|c| c.clone().pow(2)).sum::<Rational>();
                    mu[i][j] = num / den;
                    let mu_b_star: Vec<Rational> = b_star[j].iter().map(|c| mu[i][j].clone() * c).collect();
                    b_i_rational = b_i_rational.iter().zip(mu_b_star.iter()).map(|(a, b)| a - b).collect();
                }
                b_star[i] = b_i_rational;
            }
            
            if k > 1 {
                k -= 1;
            }
        }
    }
}

// --- Data Loading ---

fn load_basis_from_string(data_str: &str) -> Vec<Vec<Integer>> {
    let trimmed = data_str.trim();
    if !trimmed.starts_with("[[") || !trimmed.ends_with("]]") {
        if trimmed == "[]" { return Vec::new(); }
        panic!("String data must be in the format [[\"num1\",\"num2\"],[\"num3\",\"num4\"]]");
    }
    let inner = &trimmed[1..trimmed.len() - 1];

    inner.split("],[")
        .map(|s| {
            let row_str = s.trim_matches(|c| c == '[' || c == ']');
            row_str.split(',')
                .map(|num_str| {
                    // Trim quotes and whitespace, then parse
                    let clean_num_str = num_str.trim().trim_matches('"');
                    clean_num_str.parse::<Integer>().expect("Invalid large integer in data string")
                })
                .collect()
        })
        .collect()
}

fn load_basis_from_csv(path: &str) -> Vec<Vec<Integer>> {
    let file = File::open(path).expect("Could not open file");
    let reader = BufReader::new(file);
    let mut basis = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Error reading line");
        let row: Vec<Integer> = line
            .split(',')
            .map(|s| s.trim().parse::<Integer>().expect("Invalid number in CSV"))
            .collect();
        basis.push(row);
    }
    basis
}

// --- Main Execution Logic ---

fn run_test() {
    let mut basis: Vec<Vec<Integer>> = vec![
        vec![Integer::from(1), Integer::from(1), Integer::from(1)],
        vec![Integer::from(-1), Integer::from(0), Integer::from(2)],
        vec![Integer::from(3), Integer::from(5), Integer::from(6)],
    ];
    println!("Original basis: {:?}", basis);
    let delta = Rational::from((3, 4)); // delta = 0.75
    lll(&mut basis, &delta);
    println!("\nReduced basis (LLL): {:?}", basis);
}

fn format_basis_as_json(basis: &[Vec<Integer>]) -> String {
    let rows: Vec<String> = basis.iter().map(|row| {
        let nums: Vec<String> = row.iter().map(|num| format!("\"{}\"", num.to_string())).collect();
        format!("[{}]", nums.join(","))
    }).collect();
    format!("[{}]", rows.join(","))
}

fn main() {
    let args = Args::parse();

    if args.test {
        run_test();
        return;
    }

    let mut basis = if let Some(path) = args.file {
        load_basis_from_csv(&path)
    } else if let Some(data_str) = args.data {
        load_basis_from_string(&data_str)
    } else {
        eprintln!("❗ Please specify input with --test, --file <path>, or --data <array>");
        return;
    };

    if basis.is_empty() {
        println!("[]");
        return;
    }

    // The delta parameter for LLL, typically between (0.25, 1). 0.75 is standard.
    let delta = Rational::from((3, 4)); 
    
    lll(&mut basis, &delta);

    // Print the result in a machine-readable format
    println!("{}", format_basis_as_json(&mut basis));
}
