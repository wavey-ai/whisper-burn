use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};

use crate::{model::*, token::Gpt2Tokenizer, transcribe::spectrogram_to_text};
use burn::record::{DefaultRecorder, Recorder, RecorderError};
use burn_tch::{TchBackend, TchDevice};
use num_traits::ToPrimitive;
use std::error::Error;

pub struct TextResult {
    text: String,
}

impl TextResult {
    pub fn text(&self) -> &str {
        return &self.text;
    }
}

pub struct MelToText {
    whisper: Whisper<TchBackend<f32>>,
    tokeniser: Gpt2Tokenizer,
}

impl MelToText {
    pub fn new(model_name: &String) -> Result<Self, Box<dyn Error>> {
        let tokeniser = Gpt2Tokenizer::new().unwrap();
        let whisper_config = WhisperConfig::load(&format!("{}.cfg", model_name))?;
        let device = burn_tch::TchDevice::Cuda(0);
        let whisper =
            load_whisper_model_file::<burn_tch::TchBackend<f32>>(&whisper_config, model_name)?
                .to_device(&device);
        Ok(MelToText { whisper, tokeniser })
    }

    pub fn add(&self, mel: &Vec<f32>) -> TextResult {
        let (text, tokens) = spectrogram_to_text(&self.whisper, &self.tokeniser, mel).unwrap();
        return TextResult { text };
    }
}

fn load_whisper_model_file<B: Backend>(
    config: &WhisperConfig,
    filename: &str,
) -> Result<Whisper<B>, RecorderError> {
    DefaultRecorder::new()
        .load(filename.into())
        .map(|record| config.init().load_record(record))
}
