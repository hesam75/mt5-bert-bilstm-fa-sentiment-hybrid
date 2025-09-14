# mt5-bert-bilstm-fa-sentiment-hybrid
A hybrid pipeline for Persian (and multilingual) sentiment analysis. If an input text exceeds 150 tokens, it is first summarized with mT5; then sentiment is predicted via a BiLSTM branch (optionally fused with BERT features).
