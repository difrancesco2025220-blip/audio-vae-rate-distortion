# Rate-Distortion and Latent Stochasticity in Audio VAEs

Final project for the course **Deep Learning and Applied AI 2026**, Sapienza University of Rome.

## Research question

This project investigates how reducing the Rate of an audio Variational Autoencoder (VAE) affects reconstruction sensitivity to posterior stochasticity.

The approximate posterior is:

qφ(z|x) = N(μ(x), diag(σ²(x)))

and the β-VAE objective is:

Lβ = D + βR

where:

R = Ex KL(qφ(z|x) || p(z))

The Rate R represents the information cost of the latent representation.

We compare reconstruction using the posterior mean:

Dμ = d(x, gθ(μ))

with the expected distortion under posterior sampling:

Dsample = Ez~qφ(z|x) [d(x, gθ(z))]

and define the stochasticity gap:

Gstoch = Dsample − Dμ

The main empirical observation is that, as the Rate decreases, posterior uncertainty increases and the stochasticity gap becomes substantially larger.

## Controlled latent perturbations

To study latent stochasticity directly, we introduce the perturbation

zτ = μ + τσ ⊙ ε

with:

ε ~ N(0, I)

and τ controlling the magnitude of the stochastic perturbation.

In latent space, the following relation holds exactly:

E||zτ − μ||² = τ² Σi σi²

For small perturbations, a local linearization of the decoder predicts approximately:

E||gθ(zτ) − gθ(μ)||² ≈ c τ²

where c measures local decoder sensitivity to posterior stochasticity.

For τ ≤ 1, the quadratic approximation fits the experimental results extremely well, with R² > 0.998 for all analyzed regimes.

The sensitivity coefficient c increases strongly as the Rate decreases.

## Dataset

We use the **Free Spoken Digit Dataset (FSDD) v1.0.10**.

Main characteristics:

- 3000 WAV recordings
- sampling rate: 8 kHz
- spoken digits from multiple speakers
- 2700 training samples
- 300 test samples
- indices 0–4 are used for testing
- indices 5–49 are used for training

Audio signals are corrupted with Gaussian noise with standard deviation 0.05 and transformed into 128 × 128 log-STFT representations.

The dataset is downloaded automatically by `00_prepare_data.ipynb` and is not included in the repository.

## Model and training strategy

A convolutional VAE with latent dimension 128 is used.

Direct VAE training produced poor reconstructions, so the final pipeline uses two stages:

1. pre-training of a deterministic convolutional autoencoder;
2. transfer of its weights to the VAE, followed by gradual variational fine-tuning.

The rate-distortion analysis is performed over several values of β between 0 and 10⁻².

## Execution order

1. `00_prepare_data.ipynb`  
   Downloads and prepares the FSDD dataset.

2. `01_autoencoder_stft.ipynb`  
   Pre-trains the deterministic log-STFT autoencoder.

3. `02_rate_distortion.ipynb`  
   Trains the VAE models over the β grid and evaluates the rate-distortion trade-off.

4. `03_sampling_and_audio_metrics.ipynb`  
   Compares reconstruction using z = μ with reconstruction under posterior sampling and computes audio metrics.

5. `04_latent_analysis.ipynb`  
   Analyzes KL divergence per latent dimension, effective dimensionality, posterior mean and posterior standard deviation.

6. `05_tau_stochasticity.ipynb`  
   Performs the controlled τ perturbation experiment and fits the local quadratic sensitivity law.

7. `06_seed_robustness.ipynb`  
   Repeats VAE fine-tuning with three random seeds to evaluate robustness of the main trends.

## Main results

| β | Rate | Dμ | Dsample | Gstoch |
|---:|---:|---:|---:|---:|
| 1e-4 | 338.12 | 0.2494 | 0.2535 | 0.0041 |
| 1e-3 | 122.49 | 0.2591 | 0.2982 | 0.0391 |
| 1e-2 | 33.74 | 0.3220 | 0.4917 | 0.1697 |

The main trend is:

**lower Rate → larger posterior uncertainty → larger stochasticity gap**

The mean posterior standard deviation increases from approximately 0.229 at β = 10⁻⁴ to 0.771 at β = 10⁻².

At the same time, the stochasticity gap increases from approximately 0.0041 to 0.1697.

All 128 latent dimensions remain active under the threshold KLi > 0.01, while effective dimensionality decreases only moderately. Therefore, the reduction in Rate is not mainly explained by complete latent-coordinate shutdown.

## τ-sensitivity results

For τ ≤ 1, the decoder output displacement is approximately quadratic in τ.

The fitted sensitivity coefficient c increases as the Rate decreases:

| β | Rate | c | R² |
|---:|---:|---:|---:|
| 1e-4 | 338.12 | 0.0045 | > 0.999 |
| 3e-4 | 222.94 | 0.0129 | > 0.999 |
| 1e-3 | 122.49 | 0.0409 | > 0.999 |
| 1e-2 | 33.74 | 0.1843 | > 0.998 |

This result is consistent with the VAE reparameterization mechanism: models operating at lower Rate show stronger sensitivity to latent stochastic perturbations.

## Robustness

The main trend remains stable across three different VAE fine-tuning seeds.

The three runs always start from the same pre-trained deterministic autoencoder, so the robustness analysis concerns VAE fine-tuning rather than the complete training pipeline.

The observed association between posterior uncertainty and stochasticity gap should be interpreted as a strong empirical association consistent with the reparameterization mechanism, rather than as isolated causal evidence.

## Repository structure

- `00_prepare_data.ipynb` — dataset preparation
- `01_autoencoder_stft.ipynb` — deterministic autoencoder
- `02_rate_distortion.ipynb` — rate-distortion analysis
- `03_sampling_and_audio_metrics.ipynb` — posterior sampling and audio metrics
- `04_latent_analysis.ipynb` — latent-space analysis
- `05_tau_stochasticity.ipynb` — controlled τ perturbations
- `06_seed_robustness.ipynb` — robustness across random seeds
- `results/` — numerical outputs from the final experiments
- `report/` — final report, bibliography and report figures
- `requirements.txt` — Python dependencies
- `.gitignore` — excluded datasets, checkpoints and temporary files

## Reproducibility

The notebooks use fixed random seeds whenever possible.

The dataset and trained model checkpoints are not stored in the repository because they can be regenerated by running the notebooks in order.

Final numerical outputs used for the analysis are available in the `results/` directory.

## Environment

Tested with **Python 3.12**.

Install the required dependencies with:

`python -m venv .venv`

`pip install --upgrade pip`

`pip install -r requirements.txt`

## Report

The `report/` directory contains the final project report written using the official **DLAI 2026** template, together with the bibliography and all figures used in the report.
