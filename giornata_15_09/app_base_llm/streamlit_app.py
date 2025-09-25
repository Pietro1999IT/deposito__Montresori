"""
- Crea interfaccia streamlit per un chatbot basato su gpt tramite azure
- Imposta uno stream della risposta, la risposta viene caricata parola per parola
- Imposta due schermate, una con fino a sopra, una dove l'utente inserisce model, endpoint
  e chiave azure così l'utente può usare il chatbot con il suo modello
- Fare deploy modello embedding di azure che restituisce il vettore data una frase
- Torna su streamlit e guarda come implementare tenacity che permette di fare dei 
  richiami alle api nel caso non partisse la prima in modo automatico. 
"""

import streamlit as st
from openai import AzureOpenAI


def ping_azure_openai(api_key, endpoint, deployment, api_version):
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        # Minimal test request
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

        return True if response else False
    except Exception as e:
        st.error(f"Errore durante la chiamata a openai: {e}")
        return False


# Inizializza lo stato della sessione: false no accesso
if 'access' not in st.session_state:
    st.session_state.access = False
 
# variabile true va nel chatbot
if st.session_state.access:
    st.title("chatbot")
 
    client = AzureOpenAI(
    api_version=st.session_state.api_version,
    azure_endpoint=st.session_state.endpoint,
    api_key=st.session_state.api_key,
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
 
        response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=st.session_state.deployment
        )
 
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.choices[0].message.content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
   
else:
    # non è riuscito a loggarsi, rimetti le credenziali
    st.title("inserisci credenziali")
    st.session_state.endpoint = st.text_input("Insert Azure endpoint")
    st.session_state.api_key = st.text_input("Insert Azure api key")
    st.session_state.api_version = "2024-12-01-preview"
    st.session_state.deployment = "gpt-4o"

    if st.button("Conferma"):
        if ping_azure_openai(api_key=st.session_state.api_key, 
                              deployment=st.session_state.deployment,
                              endpoint=st.session_state.endpoint, 
                              api_version=st.session_state.api_version):
            
            st.session_state.access = True
            st.rerun()
