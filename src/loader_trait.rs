pub trait LoaderTrait {
    fn sample(&self, percentage: f32) -> Self;

    fn random_sample(&self, percentage: f32) -> Self;
}
