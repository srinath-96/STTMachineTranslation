import pandas as pd
import json

def create_simple_dataset(conversation_titles_path, conversations_path, output_path):
    """
    Creates a dataset with a simple prompt format, inspired by the successful Llama 2 script.
    """
    try:
        # Load and merge data as before
        titles_df = pd.read_csv(conversation_titles_path)
        convs_df = pd.read_csv(conversations_path)
        merged_df = pd.merge(convs_df, titles_df, on='date')

        # Combine titles and conversation sentences
        titles_data = merged_df[['kor_title', 'eng_title']].rename(
            columns={'kor_title': 'korean', 'eng_title': 'english'}
        )
        convs_data = merged_df[['kor_sent', 'eng_sent']].rename(
            columns={'kor_sent': 'korean', 'eng_sent': 'english'}
        )
        
        full_df = pd.concat([titles_data, convs_data], ignore_index=True)
        full_df.drop_duplicates(inplace=True)
        full_df.dropna(inplace=True) # Drop rows with any missing values
        
        print(f"Created a combined dataset with {len(full_df)} unique examples.")

        # Define the new simple prompt template
        prompt_template = (
            "### Translate Korean to English:\n"
            "{korean}\n\n"
            "### Translation:\n"
            "{english}"
        )

        # Create the JSONL file with just a single "text" field
        with open(output_path, 'w', encoding='utf-8') as f:
            for index, row in full_df.iterrows():
                text = prompt_template.format(korean=row['korean'], english=row['english'])
                json_record = {"text": text}
                f.write(json.dumps(json_record) + '\n')

        print(f"\nSuccessfully created simple dataset at '{output_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Execution ---ß
if __name__ == '__main__':
    conversation_titles_file = 'conversation_titles.csv'
    conversations_file = 'conversations.csv'
    output_file = 'gemma_simple_dataset.jsonl' # New file name
    
    create_simple_dataset(conversation_titles_file, conversations_file, output_file)