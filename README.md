# Forged-GAN-BERT
Forged-GAN-BERT: Authorship Attribution for LLM-Generated Forged Novel

This repository contains the code and data used for our [EACL SRW paper][1].

If you use these resources, please cite:

<b>Forged-GAN-BERT: Authorship Attribution for LLM-Generated Forged Novels</b>. Kanishka Silva, Ingo Frommholz, Burcu Can, Fred Blain, Raheem Sarwar, Laura Ugolini (2024).

    @inproceedings{silva-etal-2024-forged,
      title = "Forged-{GAN}-{BERT}: Authorship Attribution for {LLM}-Generated Forged Novels",
      author = "Silva, Kanishka  and
        Frommholz, Ingo  and
        Can, Burcu  and
        Blain, Fred  and
        Sarwar, Raheem  and
        Ugolini, Laura",
      editor = "Falk, Neele  and
        Papi, Sara  and
        Zhang, Mike",
      booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
      month = mar,
      year = "2024",
      address = "St. Julian{'}s, Malta",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2024.eacl-srw.26",
      pages = "325--337",
      abstract = "The advancement of generative Large Language Models (LLMs), capable of producing human-like texts, introduces challenges related to the authenticity of the text documents. This requires exploring potential forgery scenarios within the context of authorship attribution, especially in the literary domain. Particularly,two aspects of doubted authorship may arise in novels, as a novel may be imposed by a renowned author or include a copied writing style of a well-known novel. To address these concerns, we introduce Forged-GAN-BERT, a modified GANBERT-based model to improve the classification of forged novels in two data-augmentation aspects: via the Forged Novels Generator (i.e., ChatGPT) and the generator in GAN. Compared to other transformer-based models, the proposed Forged-GAN-BERT model demonstrates an improved performance with F1 scores of 0.97 and 0.71 for identifying forged novels in single-author and multi-author classification settings. Additionally, we explore different prompt categories for generating the forged novels to analyse the quality of the generated texts using different similarity distance measures, including ROUGE-1, Jaccard Similarity, Overlap Confident, and Cosine Similarity.",
}

[1]: [https://aclanthology.org/2023.acl-srw.44/](https://aclanthology.org/2024.eacl-srw.26/)https://aclanthology.org/2024.eacl-srw.26/
