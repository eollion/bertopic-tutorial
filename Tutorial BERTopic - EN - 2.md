When working with a new corpus, the first step is to explore the documents in order to gain a sense of what they contain and what they don't. This necessary step of exploration can be done by hand, reading each document, but when dealing with large corpora, this task is almost impossible; hence, for quite some time now, social scientists have used **topic modelling** techniques to quickly extract the main themes in their corpus (Asmussen & Møller, 2019). 

> **Topic modelling** is a natural language processing task that extracts latent topics structuring a corpora.
> *For instance, Jockers & Mimno (2013) extracted broad themes in the English literature from the 19th century.*
> **Natural language processing (NLP)** is a subfield of computer science that analyses textual data. Main NLP tasks are text generation (like chatGPT), text classification, or topic modelling.

![[mermaid-diagram-2025-10-28-161840.png#center|500]]
Until recently, topic modelling techniques — and natural language processing (NLP) in general — heavily relied on word counts. Among other limits, word counts do not take into account the context in which a word is used, thus failing to grasp the complexity of the human language. 
To tackle this limit, researchers developed transformer models that generate richer and denser representations; these representations are vectors called embeddings. BERTopic is the new topic model that leverages transformer techniques to generate coherent topics based on the semantic similarity of texts. BERTopic has already proven to be useful in social science contexts. *For instance, Bizel-Bizellot et al. (2024) used it to identify the main circumstances of infection with COVID-19 based on survey-free-text answers.*

BERTopic is a brilliant tool, but it requires time to master. In this tutorial, we break down the whole architecture for someone with a social science background to understand the underlying technology. We will demonstrate how to use and tune a topic model to your needs. For the sake of the demonstration, we will use the database of all the theses defended between 2010 and 2022 in France and create a topic model to describe what keeps French PhD students busy!
By the end of this tutorial, you should be able to: 
- Get an idea of what you can do with topic modelling in social science
- Set up a topic model on your data and make sense of the results
- Understand each step of the BERTopic pipeline and customise it
We conclude this tutorial with a discussion about the evaluation of topic modelling and good practices for reproducibility.
# Python, Machine Learning and NLP prerequisites
## Python
We won't cover how to install the coding environment. For this, please refer to **XXX**.
We assume that:
- you have a working environment
- you are able to install packages
- you know the basic syntax of Python (functions, variables, if-statements, for loops) and you're comfortable enough with Pandas to load your documents and proceed to simple manipulations such as creating, dropping and renaming columns and rows.
Packages to install: 
```python
pip install pandas bertopic
```
*nota: we provide a detailed requirement file that should work for macs and linux*
## Machine Learning and NLP 
We don't assume that you have any knowledge about NLP and try our best to explain every step in an agnostic manner. We also provide numerous references if you want to dig deeper.
# Material
The tutorial comes with some material uploaded on [Zenodo](https://doi.org/10.5281/zenodo.17416954) :
- a notebook with all the code
- The original dataset (which can be downloaded [here](https://www.data.gouv.fr/datasets/theses-soutenues-en-france-depuis-1985/)).
- A clean dataset with the cleaning code.
# The BERTopic pipeline
## What we expect
The BERTopic pipeline takes a list of text documents and returns meaningful topics as well as a mapping from the text documents to the said topics. The goal is to retrieve semantically coherent clusters and describe your corpus with different topics and keywords. 
Here is a basic usage of BERTopic: 
```python
from bertopic import BERTopic

# Load your documents
documents = [
	"My cat is the cutest.",
	"Offer your cat premium food.",
	"The Empire State Building is 1,250 feet tall",
]

# Create a BERTopic object
topic_model = BERTopic()
# Fit your model to your documents
topic_model.fit(documents)
# Predict the topics and probabilities
topic, probabilities = topic_model.transform(documents)

# Or do it all at once
topic, probabilities = topic_model.fit_transform(documents)
```
*Nota: the `model.fit`, `model.transform` is a common syntax in machine learning first introduced by Scikit-learn, one of the first and most complete machine learning library in Python* **(à vérifier)**
The `topic` variable is a list containing integers: for each document, the integer represents the topic/group it belongs to. In our case, `topic = [0, 0, 1]` as the first 2 documents talk about cats, whereas the last document is about the Empire State Building.
The `probabilities` variable is a list of floats: for each document, the float represents how close it is to the topic. 
Then we can retrieve topic information that will return keywords that best represent our corpus:
```python
topic_info = topic_model.get_topic_info()
```
The `topic_info` variable is a table like this : 

| Topic | Count | Name       | Representation | Representative_Docs                            |
| ----- | ----- | ---------- | -------------- | ---------------------------------------------- |
| 0     | 2     | 0_cat      | "cat"          | "My cat is the cutest"                         |
| 1     | 1     | 1_building | "building"     | "The Empire State Building is 1,250 feet tall" |
The `Topic` column lists the topic IDs, the `Count` column lists the number of element there are in each topic, the `Name` column is a summary of topic ID and keywords — listed in the `Representation` column) and finally the `Representative_Docs` lists example of documents that are representatives of the topic.
*Nota: this example would not run because there are not enough documents!*

At the end of the tutorial you'll be able to generate a detailed topic analysis of the thesis defended in France. Here is a preview of the results: 

| Topic | Count | Name      | Representation |
| ----- | ----- | --------- | -------------- |
| 0     | 500   | 0_**XXX** | **XXXX**       |
| 1     | 500   | 1_**XXX** | **XXXX**       |
| 2     | 500   | 2_**XXX** | **XXXX**       |

![[2d_plot.png]]
## What happens under the hood? 
Under the hood, BERTopic does the following: 
- Generates mathematical representation that will capture the semantic properties of each document — the embeddings.
- Based on the embeddings, it will split the documents into groups that are semantically close (this is called clustering). The hope is that these groups represent latent topics of our corpus.
- For each topic, it will retrieve keywords that best describe the specificity of each topic.
### How to generate the embeddings?
To generate the embeddings we use decoder models. Decoder models are a type of transformer whose job is to encapsulate the semantic of textual data. A good example of decoder is the BERT model and all it's successors like RoBERTa, or DeBERTa ...  For generating embeddings, BERTopic uses `SBERT`. The only parameter is to chose the model to use for encoding our documents. **develop + context window**

The embeddings — ie the generated vectors, contain hundreds of dimensions (for instance, the dimension of BERT's embeddings is 512). Clustering algorithms work poorly with this many dimensions[^2] so we need to reduce the dimensionality of the embedding space (tipically between 2 and 10). To reduce the dimensionality, the BERTopic pipeline uses the UMAP algorithm for it's ability to grasp local and global structures (McInnes et al., 2018)[^1]. This mean that, despite moving from several hundreds of dimension to only a couple, documents that are close together will stay close and distant ones will stay further apart. This is a critical step as we are heavily changing the structure of the data.

> [!question] Pour en apprendre plus sur les techniques de plongement.
> On vous renvoie vers ...
> - [Lien vers la documentation complète de BERTopic](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html)

---
**move further down**
The main UMAP parameters are: 
- `n_neighbors`: This is the algorithm's most important parameter. To focus on local structures, you need to choose a small `n_neighbors` and increase it's value to grasp more global values[^3]. Because of it's design, the number of neighbours depends on the number of documents you have, meaning that if you have 1,000 documents, `n_neighbors = 100` will create a very general structure, whereas if you have 100,000 documents, the same value will create a fine grained structure.
- `min_dist`: This parameter is "essentially aesthetic" (McInnes et al., 2018, p23). For clustering tasks you want to keep this value low so that the data is aggregated in dense groups that will be easier to cluster.
- `n_components`: This parameter defines the number of dimensions of the output space. We generally choose between 2 and 5 for easier visualisations. This is more difficult to tune as you'll see the effects through the clustering results. We recommend to use `n_components = 5` and modify[^4] the value if tuning the clustering parameters is not enough. 
- `metrics`: This parameter defines the metric used to quantify the similarity between 2 vectors. For NLP tasks, we use the `cosine` metric
--- 
> [!question] Learn more about dimensionality reduction and UMAP
> On dimensionality reduction:
> - XX
> 
> On UMAP: 
> - [Youtube videos to understand the main ideas (StatQuest)](https://www.youtube.com/watch?v=eN0wFzBA4Sc) and the [mathematical details (StatQuest)](https://youtu.be/jth4kEvJ3P8?si=ZM66Ko6TyV4Vyy7E).
> - [To explore the impact of the UMAP parameters](https://pair-code.github.io/understanding-umap/).
### How to generate clusters
To goal for the clustering algorithm is to create groups of documents that are semantically close. We are not certain that the output clusters will be topics, but we tune the BERTopic pipeline in order for the clusters to be representatives of topics that are latent in our corpus.

HDBSCAN was chosen for it's ability to detect cluster for various shape and density. HDBSCAN also allows for documents to be labeled as noise. This allows to focus on dense and coherent.

---
**Move further down**
The main HDBSCAN parameters are:
- `min_cluster_size`: minimum number of elements in a group to be considered a cluster, otherwise, it's considered as noise. Small `min_cluster_size` will create many, highly specify clusters whereas large `min_cluster_size` will create very general cluster as well as a lot of noise. The default value is 5. [*source 1*](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size), [*source 2*](https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#min-cluster-size)
-  `min_samples`: the number of samples in a neighbourhood for a point to be considered as a core point, ie the larger `sample_size` is, the more the algorithm will focus on dense areas.  If no value is provided, then it is set to the value of `min_cluster_size`. [*source 1*](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples) [*source 2*](https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#min-samples)
In our experience, tuning `min_cluster_size` and leaving `min_samples = min_cluster_size` is enough for most cases. 
- `metric`: here the metric should not me switched to `cosine` because we are not in the embedding space anymore. 
---
> [!question] Learn more about clustering and HDBSCAN
> On clustering:
> - XXX
> 
> On HDBSCAN:
> - [Presentation of HDBSCAN by John Healy - PyData NYC 2018](https://youtu.be/dGsxd67IFiU?si=18wnb1nh1oJxyHzH)
> - [The HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/index.html)
> - [The Scikit-Learn documentation of HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)
### How to retrieve relevant keywords for each topic 
Once we have created groups of documents, we need to create a meaningful representation with keywords. To do that, we come back to the text document and use word-count-based techniques that will count the number of occurrences of each word[^5] in each document. However, there are plenty of words we don't want to count because they do not carry much semantic information (ex: "the", "I", "is", "but", ... ). These words are called stop words and are skipped.
To do so, we use the `CountVectorizer` object that will create a word x document matrix. 

> Example: given the two following documents:
> - "My cat is the cutest.",
> - "Offer your cat premium food.",
> The word x document matrix would be:

|         | doc 1 | doc 2 |
| ------- | ----- | ----- |
| cat     | 1     | 1     |
| cutest  | 1     | 0     |
| offer   | 0     | 1     |
| premium | 0     | 1     |
| food    | 0     | 1     |


The word x document matrix is never used as is and needs transformation to outline relevant words, ie words that appear often enough to be representative, but not so much. The usual transformation is called TF-IDF. This transformation raises the score of words that appear often in a document and decreases the score of words that appear in many documents. 
In BERTopic, they use an alternative called **c-TF-IDF**. This transformation raises the score of words that appear often in documents of the same group and decreases the score of words appearing in other groups. With this transformation, we retrieve words that make a group unique!
## The whole pipeline together
We described each step of the pipeline that is illustrated in the documentation.
![[bertopic-general-en.svg#center|500]]
To sum it up we have: 
- First, generate embedding that encapsulate semantic information with **SBERT** and reduce the dimensionality of vectors to a manageable number of dimensions with **UMAP**.
- Then create groups of semantically close documents with **HDBSCAN**. Each group can represent latent topics in our corpus.
- Create meaningful representations of each topics by counting words in the documents with **CountVectorizer** and outline more representative words with **c-TD-IDF**.
# Hands on with Python
> [!attention] Comments from Emilien
> **LA QUALIT2 Du TOPIC ANALYSIS DEPEND DU PRETRAITEMENT**
> **PCQ ENCODEUR ON ENLEVE PAS LES STOP WORDS** 

As mentioned before, we will use the dataset listing all theses defended in France since 1985. The original dataset can be downloaded on [data.gouv.fr](https://www.data.gouv.fr/datasets/theses-soutenues-en-france-depuis-1985/). To avoid excessive pre-processing, we curated the dataset and uploaded it (with the code) on [Zenodo](https://doi.org/10.5281/zenodo.17416954). 

> [!NOTE]+ The curations are listed below:
> - select only thesis defended between 2010 and 2022 **propose an explanation**
> - select thesis where both resumes in english and french exist, as well as the oai code (correspond to the field of the thesis, see [oai_codes.csv in the Zenodo project]). 
  (about 37% of the dataset remains, representing about 166k rows).
> - aggregate the topics (*sujets rameaux*) together under a single column (previously 53 (!!!))
> - check that the provided resume under the english column is written in english, and the resumes under the french column are written in french. If the english version is in the french column and vice-versa, swap them.
> - Finally, only select the rows where the resumes in english are in english and the resumes in french are in french. (36% of the original dataset, representing about 164k rows)
> - Add an index to the dataset

The preprocessing step is **the most important step**. Although we can tune the topic modelling towards meaningful clusters and representations, your corpus is your input and no model will generate good results out of poor input. We list a number of questions you need to consider and justify for your topic model to be relevant to your research: 
- *Is my corpus homogenous enough ?*
  It could be tempting to shove millions of different documents from different sources in a topic model and see what comes out. However, to make sure that the groups will represent topics, one must be sure that your documents are similar in formality, tone, length, density of information etc... If your corpus is too heterogenous, the topic model can highlight theses differences and you will lose sight of meaningful latent topics.[^6]
  In our case, as we analyse theses resumes which are quite standardised, the corpus should be homogenous enough for the topic model to pick up topics and not other semantic dimensions.
- *Are my documents in the right language?*
  Most of the time, language models are trained in a single language. Some models are said multi-lingual and accept texts in more than one language. However, in our experience, working documents in different languages generates poor topics as the language shift holds for the most salient difference and each language is clustered by itself. We recommend to translate your documents in a single language.
  In our case, the data curation allowed us to extract theses where both the english and french resumes were provided.
- *How long are my documents?*
  One need to precisely define their task before diving into topic modelling. What are you trying to analyse ? Will this information be available at the sentence level ? paragraph level ? the document level ? 
  In our case, the topic of the thesis will be described throughout the resume, hence the resumes must  be taken as a whole and not subdivided at the sentence level.
  
  Also, as introduced before, each embedding model has a context window, meaning that long documents will be truncated. One must make sure that the length of the documents in their corpus is smaller than the model's context window. If the context window is too small consider changing embedding model. Careful though, larger context window means longer computation time and greater computation resources required to run the model.
  We will check the length of our documents before.

Let's load the dataset:
```python
df_raw = pd.read_csv("path/to/theses-soutenues-clean-with-index.csv")
```

The dataset contains the following columns :
- *CI*: Custom index, values are `CI-XXXX`, with `XXXX` ranging from 0 to 164,378
- *year*: the year of the defence, values are integers ranging from 2010 to 2022
- *oai_set_specs*: the oai code, each code looks like `ddc:XXX`, for instance `ddc:300` refers to `Sciences sociales, sociologie, anthropologie`.
- *resumes.en* and *resumes.fr*: the resumes of the thesis, respectively in english and french. We are sure that every row contains a valid resume in the right language thanks to the data curation.
- *titres.en* and *titres.fr*: the titles of the thesis, respectively in english and french. Only 5% of the rows do not have a valid title (french or english). The language of the titles has not been checked because it will only be used to check the qualitative validity of topic model. [Refer to the notebook to see the code.](google.com)
- *topics.en* and *topics.fr*: the aggregated topics provided by the author. Only 5% of the rows do not have valid topics (french or english). The language of the topics has not been checked because they will only be used to check the qualitative validity of topic model.

Let's take some time to check if our documents fit inside the context window.
To retrieve the context window size, you can check the hugging page of the model or load the config as such:
```python
config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
print(f"Context window size of the model {model_name}: {config.max_position_embeddings}")
```
Let's look at two models, `sentence-transformers/all-MiniLM-L6-v2` (default embedding model in the BERTopic pipeline) and `Alibaba-NLP/gte-multilingual-base`.
```bash
>>> Context window size of the model sentence-transformers/all-MiniLM-L6-v2: 512
>>> Context window size of the model Alibaba-NLP/gte-multilingual-base: 8192
```
And now let's look at the length of our documents:

```python
df_raw["resumes.en.len"] = df_raw["resumes.en"].apply(len)
df_raw["resumes.fr.len"] = df_raw["resumes.fr"].apply(len)
df_raw.loc[:,["resumes.en.len", "resumes.fr.len"]].describe()
```

|      | resumes.en | resumes.fr |
| ---- | ---------- | ---------- |
| min  | 1          | 6          |
| 25%  | 1324       | 1508       |
| 50%  | 1617       | 1702       |
| 75%  | 2080       | 2362       |
| max  | 12010      | 12207      |
| mean | 1777       | 1984       |
| std  | 735        | 802        |
With these statistics, we see that we can rule out using `sentence-transformers/all-MiniLM-L6-v2` because it's context window is too narrow. By keeping resumes between 1000 and 4000 characters (ie between 300 and 1300 tokens[^7]) we can keep most of the dataset (89%).
```python
valid_index = logical_and.reduce([
	df_raw["resumes.fr.len"] >= 1000,
	df_raw["resumes.fr.len"] <= 4000,
	df_raw["resumes.en.len"] >= 1000,
	df_raw["resumes.en.len"] <= 4000,
])

df = df_raw.loc[valid_index,:]
```
To avoid extra computation time, we are going to sample a small number of documents and stratify this sampling by the year of the defence[^8]. 
```python
stratification_column = "year"
samples_per_stratum = 500
df_stratified = (
	df
	.groupby(stratification_column, as_index = False)
	.apply(lambda x : x.sample(n = samples_per_stratum), include_groups=True)
	.reset_index()
	.drop(["level_0", "level_1"], axis = 1)
)
# Save the preprocessed dataset
df_stratified.to_csv("path/to/theses-preprocessed.csv", index=False)
```
The resulting stratified DataFrame contains 6500 rows.
## Create a BERTopic instance, fit and  transform
To create a `topic_model` object we need to use the `BERTopic` constructor and define some parameters. For now, we will not change the default parameters of the clustering model (`hdbscan_model`) or the dimension reduction model (`umap_model`). We will however define the language of the corpus as well as the vectorizer model in order to remove all stopwords[^11] and retrieve meaningful topics. Then, one must use the `fit` method to fit the topic model to the corpus. We could ask BERTopic to embed the documents, but because we already have these embeddings, we just need to passe them on.
```python
language = "english" # or "french"
language_short = language[:2] # "en" or "fr"
vectorizer_model = CountVectorizer(
	stop_words = list(stopwords(language_short)),
	ngram_range = (1,1)
)

topic_model = BERTopic(
	language = language,
	vectorizer_model = vectorizer_model,
)
topic_model.fit(documents=docs, embeddings=embeddings)
```
# Discussion on good practices
## Reproducibility
Pre-computing the embeddings is a good practice as it will prevent from computing them at each run, but also because it allows you to use a broader spectrum of embedding models that could ne necessarily be used with BERTopic[^7].
We retrieve the embeddings and the documents
```python
ds = load_from_disk("path/to/file")
docs = np.array(ds[f"resumes.en"]) # Number of documents : 6500
embeddings = np.array(ds["embedding"]) # shape : (6500, 768)
```

The columns in the dataset are the same as before in addition to an `embedding` column containing the embeddings of the resumes.
### Save your instance locally
For reproducibility purposes, and more generally, to save your work, BERTopic lets you do that with the `save` method. Two parameters of importance:
- `serialization (str)`: must be `"safetensors"`, `"pickle"` or `"pytorch"`. We recommend to use `"safetensors"` or the `"pytorch"` format as they are broadly used in machine learning and recommended by the [BERTopic documentation](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.save)[^6].
- `save_ctfidf (bool)` : wether to save the vectorizer configuration or not. This is the heaviest bit (see table below).  
```python
# ~ 500 KB
topic_model.save(
	path = "./bertopic-default",
	serialization = "safetensors",
	save_ctfidf = False
)

# ~ 6MB
topic_model.save(
	path = "./bertopic-default-with-ctfidf",
	serialization = "safetensors",
	save_ctfidf = True
)
```
*à cacher - pas en première lecture -> faire une partie reproducibilité*

| filename (size)                                                              | definition                                                                                                                                                                                                                        | `save_ctfidf = False` |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `config.json`(~0.3 KB)                                                       | parameters of necessary to recreate the BERTopic instance                                                                                                                                                                         | ✅                     |
| `topics.json`(~150 KB)                                                       | file containing the topic representations (keywords), a list with the topics associated to each document, topic sizes, mapper[^5] and labels                                                                                      | ✅                     |
| `topic_embeddings.safetensors` (~350 KB) or `topic_embeddings.bin` (~450 KB) | the embeddings of the centroids of each topic.<br>shape : $(n_{topics}\times dim_{embeddings})$                                                                                                                                   | ✅                     |
| `ctfidf_config.json` (~2.1 MB)                                               | The configuration to recreate the weighting scheme object (`ClassTfidfTransformer`) and the vectorizer model (`CountVectorizer`) including the stop words and the vocabulary with the number of occurrences for for each element. | ❌                     |
| `ctfidf.safetensors` (~4 MB) or `ctfidf.bin` (~4 MB)                         | the c-TF-IDF representations, a topic x n-grams table.<br>shape: $(n_{topics}\times n_{n-grams})$                                                                                                                                 | ❌                     |
*Note: the sizes provided correspond to a vectorizer model only counting unigrams. If you account for the bigrams, the size of the files grows exponentially.*

The to reload your instance you just need to use the `load` method: 
```python
topic_model = BERTopic.load("./bertopic-default")
```

Saving the instance is a good practice, as we will see below, when reducing the number of topics, the instance is updated and you can't go back. Hence, we would recommend to save at least one instance — *or rerun the whole cell*. 
---
[^1]: The UMAP algorithm is very close to the t-SNE algorithm with better scaling capabilities.
Another good option for the dimensionality reduction step is the PCA algorithm that will focus on the global structure. PCA is a better choice if you solely focus on the big picture (McInnes et al., 2018, p45).

[^2]: For example, running the topic model with no dimensionality reduction takes about 10 times as much time and generates poor topics.

[^3]: "represents some degree of trade-off between fine grained and large scale manifold features" (McInnes et al., 2018, p23).

[^4]: Increase if you want to add more semantic features, or decrease to remove some semantic features. The effect of adding/removing a number of features is not obvious, and you'll have to try different configurations.

[^5]: To be accurate, we count for n-grams: n-grams are sequences of textual entities (tokens or words). In the context of topic modelling, 1-grams would be words, 2-grams would be sequences of 2 words and so on. 

[^6]: Disclaimer: You may want to highlight these dimensions to identify hate speech, for instance. Homogeneity has to be relevant to your use case and questioning your corpus is a part of the topic modelling pipeline that should not stay overlook. 

[^7]: Rule of thumb: 1 token = 3 characters

[^8]: More information on [stratification in Pandas](https://proclusacademy.com/blog/stratified_sampling_pandas/)
