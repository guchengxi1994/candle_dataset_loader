use std::io::Write;

use crate::dataset::Dataset;

#[test]
fn test_iter_simple_dataset() {
    let d = candle_core::Device::Cpu;
    let dataset = Dataset {
        train_data: candle_core::Tensor::ones(&[10, 2], candle_core::DType::F32, &d).unwrap(),
        train_labels: candle_core::Tensor::ones(&[10], candle_core::DType::F32, &d).unwrap(),
        test_data: None,
        test_labels: None,
        batch_size: 5,
    };
    println!("{:?}", dataset.train_data.shape());
    let _ = std::io::stdout().flush();
    dataset.into_iter().take(5).for_each(|(x, y)| {
        println!("{:?} {:?}", x.shape(), y.shape());
        let _ = std::io::stdout().flush();
    });
}

#[test]
fn test_iter_complex_dataset() {
    let d = candle_core::Device::Cpu;
    let dataset = Dataset {
        train_data: candle_core::Tensor::ones(&[10, 3, 224, 224], candle_core::DType::F32, &d)
            .unwrap(),
        train_labels: candle_core::Tensor::ones(&[10], candle_core::DType::F32, &d).unwrap(),
        test_data: None,
        test_labels: None,
        batch_size: 5,
    };
    println!("{:?}", dataset.train_data.shape());
    let _ = std::io::stdout().flush();
    dataset.into_iter().take(5).for_each(|(x, y)| {
        println!("{:?} {:?}", x.shape(), y.shape());
        let _ = std::io::stdout().flush();
    });
}
