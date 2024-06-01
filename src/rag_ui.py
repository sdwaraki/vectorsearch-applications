import os
import sys

sys.path.append('../')
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from tiktoken import get_encoding
from weaviate.classes.query import Filter
from litellm import completion_cost
from loguru import logger
import streamlit as st

from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from src.llm.prompt_templates import generate_prompt_series, huberman_system_message
from app_functions import (convert_seconds, search_result, validate_token_threshold,
                           stream_chat, load_data, get_retriever, get_reranker, get_llm)

## PAGE CONFIGURATION
st.set_page_config(page_title="Huberman Labs",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

###################################
#### SET UP APP CONFIGURATION #####
###################################

# Example models
turbo = 'gpt-3.5-turbo-0125'
# claude = 'claude-3-haiku-20240307'

reader_model_name = None
collection_name = None
data_path = '../data/huberman_labs.json'
embedding_model_path = 'put your fine-tuned model here'
###################################

## RETRIEVER
retriever = get_retriever('/Users/sumanth/Downloads/content/allminilm-finetuned-256')
# if retriever._client.is_live():
#     logger.info('Weaviate is ready!')

## RERANKER
reranker = get_reranker()

## QA MODEL
llm = get_llm()

## TOKENIZER
encoding = get_encoding("cl100k_base")

## Display properties
display_properties = ['thumbnail_url', 'content', 'title', 'guest', 'summary', 'episode_url', 'length_seconds']

## Data
data = load_data(data_path)

# creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

# best practice is to dynamically load collections from weaviate using client.show_all_collections()
available_collections = ['Huberman_minilm_128', 'Huberman_minilm_256', 'Huberman_minilm_512', 'Huberman_minilm_256_fine_tuned']

## COST COUNTER
if not st.session_state.get('cost_counter'):
    st.session_state['cost_counter'] = 0


def main(retriever: WeaviateWCS):
    #################
    #### SIDEBAR ####
    #################
    with st.sidebar:
        collection_name = st.selectbox('Collection Name:', options=available_collections, index=None,
                                       placeholder='Select Collection Name')
        guest_input = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        alpha_input = st.slider("Select alpha", 0.0, 1.0, value=None)
        retrieval_limit =st.number_input(label='Select limit', min_value=1, max_value=100, value=None, placeholder='Select retrieval limit')
        reranker_topk = st.number_input(label='select reranker top_k', min_value=1,  value=None, placeholder='Select reranker top_k')
        temperature_input = st.slider("Select Temperature", 0.0, 2.0, 0.5)
        verbosity = st.slider(label="select verbosity", min_value=0, max_value=2, value=1, step=1)

    # retriever.return_properties.append('expanded_content')

    ##############################
    ##### SETUP MAIN DISPLAY #####
    ##############################
    st.image('./app_assets/hlabs_logo.png', width=400)
    st.subheader("Search with the Huberman Lab podcast:")
    st.write('\n')
    col1, _ = st.columns([7, 3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')
    #######################
    #### SEARCH + LLM #####
    #######################
    if query and not collection_name:
        raise ValueError('Please first select a collection name')
    if query:
        # make hybrid call to weaviate
        guest_filter = Filter.by_property(name='guest').equal(guest_input) if guest_input else None

        hybrid_response = retriever.hybrid_search(filter=guest_filter,
                                                  request=query,
                                                  collection_name=collection_name,
                                                  return_properties=display_properties,
                                                  alpha=alpha_input,
                                                  limit=retrieval_limit
                                                  )

        ranked_response = reranker.rerank(results=hybrid_response,
                                          query=query,
                                          top_k=reranker_topk)
        logger.info(f'# RANKED RESULTS: {len(ranked_response)}')

        token_threshold = 2500  # generally allows for 3-5 results of chunk_size 256
        content_field = 'content'

        # validate token count is below threshold
        valid_response = validate_token_threshold(ranked_response,
                                                  query=query,
                                                  system_message=huberman_system_message,
                                                  tokenizer=encoding,  # variable from ENCODING,
                                                  llm_verbosity_level=verbosity,
                                                  token_threshold=token_threshold,
                                                  content_field=content_field,
                                                  verbose=True)
        logger.info(f'# VALID RESULTS: {len(valid_response)}')
        # set to False to skip LLM call
        make_llm_call = True
        # prep for streaming response
        with st.spinner('Generating Response...'):
            st.markdown("----")
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response, verbosity_level=verbosity)
            if make_llm_call:
                with st.chat_message('Huberman Labs', avatar='./app_assets/huberman_logo.png'):
                    stream_obj = stream_chat(llm, prompt, max_tokens=250, temperature=temperature_input)
                    logger.info(stream_obj)
                    st.write_stream(
                        stream_obj)  # https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream

            # need to pull out the completion for cost calculation
            string_completion = ' '.join([c for c in stream_obj])
            call_cost = completion_cost(completion=string_completion,
                                        model=turbo,
                                        prompt=huberman_system_message + ' ' + prompt,
                                        call_type='completion')
            st.session_state['cost_counter'] += call_cost
            logger.info(f'TOTAL SESSION COST: {st.session_state["cost_counter"]}')

            ##################
            # SEARCH DISPLAY ##
            ##################
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length_seconds']
                time_string = convert_seconds(show_length)  # convert show_length to readable time string
                with col1:
                    st.write(search_result(i=i,
                                           url=episode_url,
                                           guest=hit['guest'],
                                           title=title,
                                           content=ranked_response[i]['content'],
                                           length=time_string),
                             unsafe_allow_html=True)
                    st.write('\n\n')

                with col2:
                    image = hit['thumbnail_url']
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)


if __name__ == '__main__':
    main(retriever)
