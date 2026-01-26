from app_logic import process_pdfs, get_retriever, run_agentic_rag
import os


def main():
    data_folder = "./data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Please add PDFs to the /data folder and run again.")
        return

    print("--- Processing Documents ---")
    chunks = process_pdfs(data_folder)
    retriever = get_retriever(chunks)

    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break

        answer = run_agentic_rag(user_query, retriever)
        print(f"\nFinal Answer:\n{answer}")


if __name__ == "__main__":
    main()
