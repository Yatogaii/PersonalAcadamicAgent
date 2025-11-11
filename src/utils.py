import requests
import logging

def get_html(url, filename):
    logging.debug("!!!!!Getting HTML content for URL:", url)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        #print("!!!!!Getted HTML content:", response.text)  # Print first 500 characters
        with open(f"raw_htmls/{filename}", "w", encoding='utf-8') as f:
            f.write(response.text)
        return True
    except Exception as e:
        print(e)
        return False