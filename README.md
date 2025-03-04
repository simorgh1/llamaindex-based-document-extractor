# Kb

It is a [Streamlit](https://docs.streamlit.io) application that extracts pdf documents defined in a specific directory and using [Gemini](https://aistudio.google.com/) and [llama-index](https://docs.llamaindex.ai/en/stable/examples/), one could query about the contents of data.

## Getting Started

First run the init-env shell to install the packages in a new venv

```shell
. ./init-end.sh
```

then set the google api key obtained from google ai studio.

```shell
export GOOGLE_API_KEY=yourkey
```


then run the application

```shell
streamlit run kb.py
```

