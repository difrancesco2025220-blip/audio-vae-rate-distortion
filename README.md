# Rate–Distortion e stocasticità latente nei VAE audio

Progetto individuale per **Deep Learning and Applied AI 2026**.

## Domanda di ricerca

Studiamo come la riduzione del Rate di un Variational Autoencoder audio influenzi
la sensibilità della ricostruzione alla stocasticità del posterior.

Il VAE usa

\[
q_\phi(z|x)=\mathcal N(\mu,\mathrm{diag}(\sigma^2))
\]

e l'obiettivo

\[
\mathcal L_\beta = D + \beta R,
\qquad
R = \mathbb E_x KL(q_\phi(z|x)\|p(z)).
\]

Confrontiamo

\[
D_\mu
\]

con la distorsione sotto vero sampling

\[
D_{\mathrm{sample}}
\]

e definiamo

\[
G_{\mathrm{stoch}}=D_{\mathrm{sample}}-D_\mu.
\]

Il risultato principale osservato è:

\[
R\downarrow
\quad\Rightarrow\quad
\bar\sigma\uparrow
\quad\Rightarrow\quad
G_{\mathrm{stoch}}\uparrow.
\]

Per isolare direttamente la stocasticità usiamo inoltre

\[
z_\tau=\mu+\tau\sigma\epsilon
\]

e troviamo localmente

\[
\mathbb E\|g(z_\tau)-g(\mu)\|^2 \approx c\tau^2,
\]

con un coefficiente \(c\) che cresce fortemente nei regimi a basso Rate.

## Dataset

**Free Spoken Digit Dataset (FSDD) v1.0.10**

- 3000 file WAV;
- 8 kHz;
- cifre pronunciate da più speaker;
- split del progetto: indici `0–4` test, `5–49` training;
- 2700 esempi di training e 300 di test.

Il dataset viene scaricato automaticamente dal notebook `00_prepare_data.ipynb`
e non è incluso nel repository.

## Ordine di esecuzione

1. `00_prepare_data.ipynb`  
   Download e controllo di FSDD.

2. `01_autoencoder_stft.ipynb`  
   Autoencoder deterministico in log-STFT. Salva `checkpoints/stft_autoencoder.pt`.

3. `02_rate_distortion.ipynb`  
   Addestra tutti i VAE sulla griglia di β e salva i checkpoint.

4. `03_sampling_and_audio_metrics.ipynb`  
   Calcola \(D_\mu\), \(D_{\mathrm{sample}}\), stochasticity gap e metriche audio.

5. `04_latent_analysis.ipynb`  
   KL per dimensione, dimensionalità effettiva, \(\sigma\) e analisi del posterior.

6. `05_tau_stochasticity.ipynb`  
   Sweep di \(\tau\) e fit quadratico della sensibilità del decoder.

7. `06_seed_robustness.ipynb`  
   Robustezza su tre seed del fine-tuning VAE.

## Struttura

```text
.
├── 00_prepare_data.ipynb
├── 01_autoencoder_stft.ipynb
├── 02_rate_distortion.ipynb
├── 03_sampling_and_audio_metrics.ipynb
├── 04_latent_analysis.ipynb
├── 05_tau_stochasticity.ipynb
├── 06_seed_robustness.ipynb
├── results/
│   └── reference_*.csv
├── report/
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   ├── dlaiml2026.sty
│   └── figures/
├── requirements.txt
└── .gitignore
```

## Risultati di riferimento

I CSV `results/reference_*.csv` contengono i numeri utilizzati nel report finale.
Sono forniti come riferimento per confrontare una nuova esecuzione del codice.

Valori chiave:

| β | Rate | D_mu | D_sample | Gap |
|---:|---:|---:|---:|---:|
| 1e-4 | 338.12 | 0.2494 | 0.2535 | 0.0041 |
| 1e-3 | 122.39 | 0.2592 | 0.2983 | 0.0391 |
| 1e-2 | 33.74 | 0.3220 | 0.4917 | 0.1697 |

Nello sweep in \(\tau\), per \(\tau\le1\), i fit di
\(\mathbb E\|g(z_\tau)-g(\mu)\|^2 \approx c\tau^2\)
hanno \(R^2>0.998\).

## Riproducibilità

I notebook fissano i seed quando possibile. La verifica multi-seed riguarda
il fine-tuning VAE a partire dallo **stesso autoencoder pre-addestrato**:
non è un retraining indipendente dell'intera pipeline.

I checkpoint e il dataset non sono versionati in Git perché possono essere rigenerati.

## Ambiente

Testato con Python 3.12. Installazione:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Su Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Report

La cartella `report/` contiene il report nel **template ufficiale DLAI 2026**,
insieme alle figure e alla bibliografia.
