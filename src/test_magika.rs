#[cfg(test)]
mod magika_tests {
    use std::collections::HashMap;
    use std::fs::{self, File};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::PathBuf;

    use candle_core::{Device, Tensor};
    use candle_nn::{layer_norm, loss, Optimizer};
    use candle_nn::{
        ops::{self},
        LayerNormConfig, Module, VarBuilder, VarMap,
    };
    use once_cell::sync::Lazy;
    use rand::Rng;
    use std::fmt::{self, Display as D};

    static CLASSES: Lazy<HashMap<String, u8>> = Lazy::new(|| {
        let mut h = HashMap::new();
        h.insert("png".to_string(), 0);
        h.insert("pdf".to_string(), 1);
        h.insert("docx".to_string(), 2);
        h
    });

    pub struct MagikaFile {
        pub head: Vec<u8>,
        pub middle: Vec<u8>,
        pub end: Vec<u8>,
        pub r#type: String,
    }

    impl D for MagikaFile {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "({})", self.r#type)
        }
    }

    impl MagikaFile {
        pub fn new(path: String) -> anyhow::Result<Self> {
            println!("Reading file: {}", path);
            let mut file = File::open(&path)?;
            let mut header_buffer = vec![0; 512];
            file.read_exact(&mut header_buffer)?;

            let file_size = file.seek(SeekFrom::End(0))?;
            if file_size < 512 {
                return Err(anyhow::anyhow!("File size is too small"));
            }

            let middle_start = (file_size / 2) as i64 - 256;
            file.seek(SeekFrom::Start(middle_start as u64))?;
            let mut middle_buffer = vec![0; 512];
            file.read_exact(&mut middle_buffer)?;

            let tail_start = file_size.saturating_sub(512) as i64;
            file.seek(SeekFrom::Start(tail_start as u64))?;
            let mut tail_buffer = vec![0; 512];
            file.read_exact(&mut tail_buffer)?;

            let ext = path.split('.').last().unwrap();

            Ok(Self {
                head: header_buffer,
                middle: middle_buffer,
                end: tail_buffer,
                r#type: ext.to_lowercase(),
            })
        }

        pub fn to_dataset(&self, device: &Device) -> anyhow::Result<(Tensor, Tensor)> {
            let mut v = Vec::new();
            v.push(Tensor::from_vec(self.head.clone(), &[512], &device)?);
            v.push(Tensor::from_vec(self.middle.clone(), &[512], &device)?);
            v.push(Tensor::from_vec(self.end.clone(), &[512], &device)?);
            let xs = Tensor::cat(&v, 0)?;
            let labels =
                Tensor::from_vec(vec![*(CLASSES.get(&self.r#type).unwrap())], &[1], &device)?;
            println!("ext {} ===> labels {:?}", self.r#type, labels);
            let ys = one_hot(&labels, CLASSES.capacity(), xs.dtype())?;

            Ok((xs, ys))
        }
    }

    pub struct Magika {
        dense1: candle_nn::Linear,
        dense2: candle_nn::Linear,
        dense3: candle_nn::Linear,
        dense4: candle_nn::Linear,
        layer_norm1: layer_norm::LayerNorm,
        layer_norm2: layer_norm::LayerNorm,
    }

    impl Magika {
        pub fn new(vb: VarBuilder, output_size: usize) -> anyhow::Result<Self> {
            let dense1 = candle_nn::linear(257, 128, vb.pp("dense1"))?;
            let dense2 = candle_nn::linear(512, 256, vb.pp("dense2"))?;
            let dense3 = candle_nn::linear(256, 256, vb.pp("dense3"))?;
            let dense4 = candle_nn::linear(256, output_size, vb.pp("dense4"))?;
            let cfg: LayerNormConfig = LayerNormConfig::default();
            let layer_norm1 = candle_nn::layer_norm(512, cfg, vb.pp("layer_norm1"))?;
            let layer_norm2 = candle_nn::layer_norm(256, cfg, vb.pp("layer_norm2"))?;

            Ok(Self {
                dense1,
                dense2,
                dense3,
                dense4,
                layer_norm1,
                layer_norm2,
            })
        }
    }

    impl Module for Magika {
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            // one hot encoding
            let dims = xs.dims();
            let batch_size = dims[0];
            let mut batch = Vec::<Tensor>::new();
            for i in 0..batch_size {
                let x = xs.narrow(0, i, 1)?.squeeze(0)?;
                let x = one_hot(&x, 257, candle_core::DType::U8).unwrap();
                let x = x.to_dtype(candle_core::DType::F32)?;
                batch.push(x);
            }
            let xs = Tensor::stack(&batch, 0)?;
            println!("xs shape: {:?}", xs.shape());
            let xs = self.dense1.forward(&xs)?;
            println!("xs shape: {:?}", xs.shape());
            // [TODO]: shoule be `SpatialDropout`
            let xs = spatial_dropout(&xs, 0.5)?;
            let xs = xs.reshape(&[xs.dims()[0], 384, 512])?;

            let xs = self.layer_norm1.forward(&xs)?;
            let xs = ops::dropout(&xs, 0.5)?;
            let xs = self.dense2.forward(&xs)?;
            let xs = self.dense3.forward(&xs)?;
            println!("xs shape: {:?}", xs.shape());
            // GlobalMaxPooling1D
            let xs = xs.max(1)?;
            println!("xs shape: {:?}", xs.shape());
            let xs = self.layer_norm2.forward(&xs)?;
            let xs = ops::dropout(&xs, 0.5)?;
            let xs = self.dense4.forward(&xs)?;
            let xs = ops::softmax(&xs, 1)?;

            Ok(xs)
        }
    }

    fn spatial_dropout(x: &Tensor, p: f64) -> candle_core::Result<Tensor> {
        let mut rng = rand::thread_rng();
        let dim_size = x.dims()[1];
        let t = x.copy()?;
        for i in 0..dim_size {
            if rng.gen::<f64>() < p {
                t.slice_assign(
                    &[0..x.dims()[0], i..i + 1, 0..x.dims()[2]],
                    &Tensor::zeros(&[x.dims()[0], 1, x.dims()[2]], x.dtype(), x.device())?,
                )?;
            }
        }

        Ok(t)
    }

    fn one_hot(
        labels: &Tensor,
        num_classes: usize,
        r#type: candle_core::DType,
    ) -> anyhow::Result<Tensor> {
        let shape = labels.shape();
        let _one_hot = Tensor::zeros(&[shape.dims()[0], num_classes], r#type, labels.device())?;
        let mut v: Vec<Tensor> = vec![];
        for i in 0..shape.dims()[0] {
            let l = labels.get(i)?;
            let l = l.to_scalar::<u8>()?;
            let l = l as usize;
            let mut t = Tensor::zeros(&[num_classes], r#type, labels.device())?;
            t = t.slice_assign(&[l..l + 1], &Tensor::ones(&[1], r#type, labels.device())?)?;
            v.push(t);
        }

        Ok(Tensor::stack(&v, 0)?)
    }

    #[test]
    fn one_hot_test() -> anyhow::Result<()> {
        let d = Device::Cpu;
        let labels = Tensor::from_vec([0u8, 2u8, 1u8, 3u8].to_vec(), &[4], &d)?;
        println!("labels shape: {:?}", labels.shape());
        let r = one_hot(&labels, 4, candle_core::DType::U8)?;
        println!("{:?}", r.to_vec2::<u8>());
        anyhow::Ok(())
    }

    #[test]
    fn magika_test() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::ones(&[1536], candle_core::DType::U8, &device)?;
        println!("tensor shape: {:?}", tensor.shape());
        // tensor = tensor.unsqueeze(0)?;
        let double_tensor = Tensor::stack(&vec![tensor.copy()?, tensor], 0)?;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let model = Magika::new(vs.clone(), 113)?;
        let out = model.forward(&double_tensor)?;
        println!("out shape {:?}", out.shape());
        let first = out.get(0)?;
        println!("first {:?}", first.to_vec1::<f32>());
        // println!("first shape {:?}", first.shape());
        let prs = candle_core::IndexOp::i(&first.unsqueeze(0)?, 0)?.to_vec1::<f32>()?;
        println!("prs {:?}", prs);
        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();
        println!("top {:?}", top);

        anyhow::Ok(())
    }

    #[test]
    fn test_magika_file() -> anyhow::Result<()> {
        let device: Device = Device::Cpu;
        let file = MagikaFile::new(r"D:\dataset\test.pdf".to_owned())?;
        let x = file.to_dataset(&device)?;
        println!("x shape: {:?}", x.0.shape());
        println!("y shape: {:?}", x.1.shape());
        println!("y value: {:?}", x.1.to_vec2::<u8>()?);
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let t = x.0.unsqueeze(0)?;
        let model = Magika::new(vs.clone(), CLASSES.capacity())?;
        let out = model.forward(&t)?;
        println!("out shape {:?}", out.shape());

        let loss = loss::mse(&out, &x.1.to_dtype(candle_core::DType::F32)?)?;
        println!("epoch: 1, loss: {}", loss);

        anyhow::Ok(())
    }

    #[test]
    fn train_simple_model() -> anyhow::Result<()> {
        let device: Device = Device::Cpu;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let model = Magika::new(vs.clone(), CLASSES.capacity())?;
        let param = candle_nn::ParamsAdamW::default();
        let mut adam = candle_nn::AdamW::new(vm.all_vars(), param)?;

        let files = read_all_files(PathBuf::from(r"D:\dataset\"))?;
        println!("files: {:?}", files.len());
        let mut magika_files = Vec::new();
        for file in files {
            let x = MagikaFile::new(file.to_str().unwrap().to_owned())?;
            magika_files.push(x);
        }
        let mut x = Vec::new();
        let mut y = Vec::new();
        for mf in magika_files {
            let r = mf.to_dataset(&device)?;
            x.push(r.0);
            y.push(r.1.squeeze(0)?);
        }
        let rs = Tensor::stack(&y, 0)?;
        let xs = Tensor::stack(&x, 0)?;
        println!("rs shape: {:?}", rs.shape());
        println!("xs shape: {:?}", xs.shape());
        std::io::stdout().flush()?;
        for epoch in 0..10 {
            let out = model.forward(&xs)?;
            let loss = loss::mse(&out, &rs.to_dtype(candle_core::DType::F32)?)?;
            println!("epoch: {}, loss: {}", epoch, loss);
            std::io::stdout().flush()?;
            adam.backward_step(&loss)?;
        }

        vm.save("model.bin")?;

        anyhow::Ok(())
    }

    #[test]
    fn test_simple_model() -> anyhow::Result<()> {
        let device: Device = Device::Cpu;
        let mut vm = VarMap::new();
        vm.load("model.bin")?;
        let vs = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let model = Magika::new(vs.clone(), CLASSES.capacity())?;
        let file = MagikaFile::new(r"D:\dataset\test.pdf".to_owned())?;
        let x = file.to_dataset(&device)?;
        let t = x.0.unsqueeze(0)?;
        let out = model.forward(&t)?;
        let first = out.get(0)?;
        println!("first {:?}", first.to_vec1::<f32>());
        // println!("first shape {:?}", first.shape());
        let prs = candle_core::IndexOp::i(&first.unsqueeze(0)?, 0)?.to_vec1::<f32>()?;
        println!("prs {:?}", prs);
        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();
        println!("top {:?}", top);

        anyhow::Ok(())
    }

    fn read_all_files(dir: PathBuf) -> anyhow::Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.file_name().unwrap() != "test.pdf" {
                paths.push(path);
            } else if path.is_dir() {
                paths.extend(read_all_files(path)?);
            }
        }

        Ok(paths)
    }
}
