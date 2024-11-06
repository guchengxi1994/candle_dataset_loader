use crate::loader_trait::LoaderTrait;
use candle_core::Tensor;
use rand::{rngs::ThreadRng, seq::SliceRandom};

fn generate_vector(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn select_random_elements(vec: &[usize], percentage: f32, rng: &mut ThreadRng) -> Vec<usize> {
    let num_to_select = (vec.len() as f32 * percentage).round() as usize;
    let mut indices: Vec<usize> = (0..vec.len()).collect();
    indices.shuffle(rng);
    indices
        .into_iter()
        .take(num_to_select)
        .map(|i| vec[i])
        .collect()
}

fn select_random_elements_in_batch(
    vec: &[usize],
    batch_size: usize,
    rng: &mut ThreadRng,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..vec.len()).collect();
    indices.shuffle(rng);
    indices
        .into_iter()
        .take(batch_size)
        .map(|i| vec[i])
        .collect()
}

pub struct Dataset {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Option<Tensor>,
    pub test_labels: Option<Tensor>,
    pub batch_size: usize,
}

impl Iterator for Dataset {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let set = self.random_sample(None);
        return Some((set.train_data, set.train_labels));
    }
}

impl LoaderTrait for Dataset {
    fn sample(&self, percentage: Option<f32>) -> Self {
        let n_train;
        if let Some(p) = percentage {
            n_train = (self.train_data.dim(0).unwrap() as f32 * p) as usize;
        } else {
            n_train = self.batch_size;
        }
        let train_data = self.train_data.narrow(0, 0, n_train).unwrap();

        Self {
            train_data,
            train_labels: self.train_labels.narrow(0, 0, n_train).unwrap(),
            test_data: self.test_data.clone(),
            test_labels: self.test_labels.clone(),
            batch_size: self.batch_size,
        }
    }

    fn random_sample(&self, percentage: Option<f32>) -> Self {
        let train_data_size = self.train_data.dim(0).unwrap();
        let mut rng = rand::thread_rng();
        let indices = generate_vector(train_data_size as usize);
        let selected_indices;
        if let Some(p) = percentage {
            selected_indices = select_random_elements(&indices, p, &mut rng);
        } else {
            selected_indices = select_random_elements_in_batch(&indices, self.batch_size, &mut rng);
        }
        println!("Selected indices: {:?}", selected_indices);
        let selected_data = selected_indices
            .iter()
            .map(|&i| {
                self.train_data
                    .narrow(0, i as usize, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let selected_labels = selected_indices
            .iter()
            .map(|&i| {
                self.train_labels
                    .narrow(0, i as usize, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        Self {
            train_data: Tensor::stack(&selected_data, 0).unwrap(),
            train_labels: Tensor::stack(&selected_labels, 0).unwrap(),
            test_data: None,
            test_labels: None,
            batch_size: self.batch_size,
        }
    }
}
