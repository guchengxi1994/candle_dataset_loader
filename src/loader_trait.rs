pub trait LoaderTrait {
    fn sample(&self, percentage: Option<f32>) -> Self;

    fn random_sample(&self, percentage: Option<f32>) -> Self;
}
