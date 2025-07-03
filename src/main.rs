use rug::{Integer, Rational, ops::Pow};
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = "Редукция базиса решетки с использованием алгоритмов LLL или BKZ с точной целочисленной арифметикой.")]
struct Args {
    /// Путь к CSV-файлу с базисом (числа в виде строк).
    #[arg(long)]
    file: Option<String>,

    /// Запустить встроенный тестовый пример.
    #[arg(long)]
    test: bool,
    
    /// Данные в виде строки JSON: [["11","3"],["2","11"]].
    #[arg(long)]
    data: Option<String>,

    /// Использовать алгоритм LLL для редукции.
    #[arg(long, group = "algo")]
    lll: bool,

    /// Использовать алгоритм BKZ для редукции.
    #[arg(long, group = "algo")]
    bkz: bool,

    /// (Только для BKZ) Размер блока.
    #[arg(long, default_value_t = 2)]
    block_size: usize,
}

// --- Вспомогательные функции для векторов Integer ---

fn subtract_vec(v1: &[Integer], v2: &[Integer]) -> Vec<Integer> {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| Integer::from(a - b)) // Явное преобразование
        .collect()
}

fn scalar_mul(scalar: &Integer, v: &[Integer]) -> Vec<Integer> {
    v.iter()
        .map(|x| Integer::from(scalar * x)) // Явное преобразование
        .collect()
}

fn scalar_mul_rational(scalar: &Rational, v: &[Rational]) -> Vec<Rational> {
    v.iter()
        .map(|x| scalar.clone() * x.clone()) // Операции для типа Rational
        .collect()
}

fn subtract_vec_rational(v1: &[Rational], v2: &[Rational]) -> Vec<Rational> {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| a.clone() - b.clone()) // Операции для типа Rational
        .collect()
}

// --- Основные алгоритмы ---

/// Вычисляет ортогональный базис Грама-Шмидта (b_star) и коэффициенты mu.
/// Это вынесено в отдельную функцию, чтобы избежать дублирования кода в LLL и BKZ.
fn compute_gram_schmidt(b: &[Vec<Integer>]) -> (Vec<Vec<Rational>>, Vec<Vec<Rational>>) {
    let n = b.len();
    let mut b_star: Vec<Vec<Rational>> = Vec::with_capacity(n);
    let mut mu = vec![vec![Rational::new(); n]; n];

    for i in 0..n {
        let mut b_i_rational: Vec<Rational> = b[i].iter().map(Rational::from).collect();
        for j in 0..i {
            // mu[i][j] = <b_i, b*_j> / <b*_j, b*_j>
            let num: Rational = b[i].iter().zip(b_star[j].iter()).map(|(bi, bs)| Rational::from(bi) * bs).sum();
            let den: Rational = b_star[j].iter().map(|c| c.clone().pow(2)).sum();
            
            mu[i][j] = if den.is_zero() { Rational::new() } else { num / den };

            let mu_b_star: Vec<Rational> = b_star[j].iter().map(|c| mu[i][j].clone() * c).collect();
            b_i_rational = b_i_rational.iter().zip(mu_b_star.iter()).map(|(a, b)| Rational::from(a - b)).collect();
        }
        b_star.push(b_i_rational);
    }
    (b_star, mu)
}


/// LLL-редукция с использованием точной арифметики.
fn lll(b: &mut Vec<Vec<Integer>>, delta: &Rational) {
    let n = b.len();
    if n == 0 { return; }

    let (mut b_star, mut mu) = compute_gram_schmidt(b);
    
    let mut k = 1;
    while k < n {
        // Шаг 1: Size reduction
        for j in (0..k).rev() {
            let mu_kj = &mu[k][j];
            // Клонируем mu_kj перед использованием методов, которые забирают владение.
            if mu_kj.clone().abs() > Rational::from((1, 2)) {
                let q = mu_kj.clone().round();
                let q_integer = q.numer().clone();
                
                // b_k := b_k - q * b_j
                b[k] = subtract_vec(&b[k], &scalar_mul(&q_integer, &b[j]));

                // b_star[k] := b_star[k] - q * b_star[j]
                let correction_b_star = scalar_mul_rational(&q, &b_star[j]);
                b_star[k] = subtract_vec_rational(&b_star[k], &correction_b_star);

                // Обновляем коэффициенты mu для k-й строки
                for i in 0..=j {
                   let val_to_mul = mu[j][i].clone();
                   mu[k][i] -= q.clone() * val_to_mul;
                }
            }
        }

        // Шаг 2: Условие Ловаса
        let norm_b_star_k_sq: Rational = b_star[k].iter().map(|c| c.clone().pow(2)).sum();
        
        // Проверка, чтобы избежать деления на ноль, если b_star[k-1] нулевой
        if b_star[k-1].iter().all(|c| c.is_zero()) {
            k += 1;
            continue;
        }
        
        let norm_b_star_k_minus_1_sq: Rational = b_star[k-1].iter().map(|c| c.clone().pow(2)).sum();
        
        if norm_b_star_k_sq >= (delta - mu[k][k-1].clone().pow(2)) * norm_b_star_k_minus_1_sq {
            k += 1;
        } else {
            // Обмен b_k и b_{k-1}
            b.swap(k, k-1);
            // Полный пересчет GS-базиса и mu после обмена
            (b_star, mu) = compute_gram_schmidt(b);
            
            if k > 1 {
                k -= 1;
            }
        }
    }
}

/// BKZ-редукция с использованием точной арифметики.
fn bkz(b: &mut Vec<Vec<Integer>>, delta: &Rational, block_size: usize) {
    let n = b.len();
    if n == 0 { return; }

    let mut iter_count = 0;
    loop {
        iter_count += 1;
        let old_basis = b.clone();

        // Обработка блоков
        for k in 0..=(n.saturating_sub(block_size)) {
            let mut block = b[k..k + block_size].to_vec();
            lll(&mut block, delta);
            // Обновляем базис результатами из LLL для блока
            for i in 0..block_size {
                b[k + i] = block[i].clone();
            }
        }

        // Если базис не изменился после полного прохода, завершаем
        if &old_basis == b || iter_count > 10 * n { // Ограничитель для предотвращения бесконечного цикла
            break;
        }
    }
}


// --- Загрузка данных и главная функция ---

fn load_basis_from_string(data_str: &str) -> Vec<Vec<Integer>> {
    let trimmed = data_str.trim();
    if trimmed == "[]" { return Vec::new(); }
    if !trimmed.starts_with("[[") || !trimmed.ends_with("]]") {
        panic!("Строка данных должна быть в формате JSON-массива массивов строк, например [[\"1\",\"2\"],[\"3\",\"4\"]]");
    }
    
    let inner = &trimmed[1..trimmed.len() - 1];

    inner.split("],[")
        .map(|s| {
            let row_str = s.trim_matches(|c| c == '[' || c == ']');
            row_str.split(',')
                .map(|num_str| {
                    let clean_num_str = num_str.trim().trim_matches('"');
                    clean_num_str.parse::<Integer>().expect("Некорректное большое число в строке данных")
                })
                .collect()
        })
        .collect()
}

fn load_basis_from_csv(path: &str) -> Vec<Vec<Integer>> {
    let file = File::open(path).expect("Не удалось открыть файл");
    let reader = BufReader::new(file);
    reader.lines()
        .map(|line| {
            let line = line.expect("Ошибка чтения строки");
            line.split(',')
                .map(|s| s.trim().parse::<Integer>().expect("Некорректное число в CSV"))
                .collect()
        })
        .collect()
}

fn format_basis_as_json(basis: &[Vec<Integer>]) -> String {
    let rows: Vec<String> = basis.iter().map(|row| {
        let nums: Vec<String> = row.iter().map(|num| format!("\"{}\"", num)).collect();
        format!("[{}]", nums.join(","))
    }).collect();
    format!("[{}]", rows.join(","))
}

fn run_test() {
    println!("--- ЗАПУСК ТЕСТОВОГО РЕЖИМА ---");
    // ИСПРАВЛЕНИЕ: Убрано `mut`, так как `basis` не изменяется напрямую.
    let basis: Vec<Vec<Integer>> = vec![
        vec![Integer::from(19), Integer::from(2), Integer::from(32), Integer::from(41), Integer::from(28)],
        vec![Integer::from(12), Integer::from(28), Integer::from(11), Integer::from(4), Integer::from(3)],
        vec![Integer::from(1), Integer::from(5), Integer::from(6), Integer::from(2), Integer::from(44)],
        vec![Integer::from(11), Integer::from(3), Integer::from(4), Integer::from(8), Integer::from(1)],
        vec![Integer::from(10), Integer::from(15), Integer::from(21), Integer::from(31), Integer::from(9)],
    ];
    
    println!("\nОригинальный базис:\n{}", format_basis_as_json(&basis));

    let delta = Rational::from((75, 100)); // delta = 0.75
    
    // Тест LLL
    let mut lll_basis = basis.clone();
    println!("\n--- Запуск LLL ---");
    let start_lll = Instant::now();
    lll(&mut lll_basis, &delta);
    let duration_lll = start_lll.elapsed();
    println!("Редуцированный базис (LLL):\n{}", format_basis_as_json(&lll_basis));
    println!("Время выполнения LLL: {:?}", duration_lll);

    // Тест BKZ
    let mut bkz_basis = basis.clone();
    let block_size = 3;
    println!("\n--- Запуск BKZ (размер блока = {}) ---", block_size);
    let start_bkz = Instant::now();
    bkz(&mut bkz_basis, &delta, block_size);
    let duration_bkz = start_bkz.elapsed();
    println!("Редуцированный базис (BKZ):\n{}", format_basis_as_json(&bkz_basis));
    println!("Время выполнения BKZ: {:?}", duration_bkz);
}

fn main() {
    let args = Args::parse();

    if args.test {
        run_test();
        return;
    }

    // Проверяем, что указан источник данных
    if args.file.is_none() && args.data.is_none() {
        eprintln!("Ошибка: Укажите источник данных с помощью --file <путь> или --data <массив>");
        return;
    }

    // Проверяем, что выбран алгоритм
    if !args.lll && !args.bkz {
        eprintln!("Ошибка: Укажите алгоритм для выполнения с помощью --lll или --bkz");
        return;
    }

    let mut basis = if let Some(path) = args.file {
        load_basis_from_csv(&path)
    } else if let Some(data_str) = args.data {
        load_basis_from_string(&data_str)
    } else {
        unreachable!(); // Эта ветка недостижима из-за проверок выше
    };

    if basis.is_empty() {
        println!("[]");
        return;
    }

    let delta = Rational::from((75, 100)); // delta = 0.75

    if args.lll {
        lll(&mut basis, &delta);
    } else if args.bkz {
        if args.block_size > basis.len() || args.block_size < 2 {
            eprintln!("Ошибка: Размер блока для BKZ должен быть между 2 и размером базиса.");
            return;
        }
        bkz(&mut basis, &delta, args.block_size);
    }

    // Печатаем результат в формате JSON для легкого парсинга
    println!("{}", format_basis_as_json(&basis));
}
