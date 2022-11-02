# biweekly-report-4-skubalon2

Allison_VAE_GAN: In this notebook, I do a brief EDA and then compare two generative architectures-- a variational autoencoder (VAE) and a generative adversarial network (GAN)-- on celebrity faces from a subset of the CelebA dataset. While I was unable to successfully train the DC-GAN, I compare the VAE model with a DC-GAN trained on the same dataset in a post found online. I got experience exploring both VAE and GAN architectures and noticed that the GAN performed better overall.


Simon-and-Team_MEME_GENERATOR: This notebook is an extention of an idea that we had last biweekly report but realized that the content we needed would be learning in this next interval of class. We follow a very mature (and somwhat hard to pull working components onto our personal computers) "DeepHumor" github to create this meme generator. The models we are integrating are transformers and LSTM, and to describe what we are loading into our code we went deep into the github to find the tensorboard outline of these models to help us with our understanding and descriptions. At the end of this notebook we have funny memes from classic meme images and images of ourselves!

Simon_Multimodal_Few_Shot: In this notebook I exectue an extension of my interest in few-shot learning to accomplish image captioning. In the tutorial that I followed for this this notebook, I was able to again utilize the powerful tool of OpenAi's GPT2 tool I was able to caption images alongside perform similar tasks such as giving a knowledgable fact about whatever content is in the input image. Another useful application of GPT2 that has made me more excited about the tool and more versitile with applying its abilities.

readingComprehensionQuestionAnsweringBot: 
In this notebook, we will see how to fine-tune one of the hugging face Transformers models (in our case we pulled down "distilbert-base-uncased") to the downstream task of question answering. Specifically with the SQuAD datasets, this is the task of extracting the answer to a question from a given context. Note that this model does not generate new text! Instead, it selects a span of the input passage as the answer.
