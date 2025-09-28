import requests
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Pare HTML content
class PareHTML:
    def __init__(
        self,
        url: str = None
    ):
        # Access website
        response = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page, 'html.parser')

        # Initialize
        content = ""

        # Retrieve title from 'meta property="og:title"'
        title = soup.find("meta", property="og:title")
        if title != None:
            content += "Title: " + title["content"] + "\n\n"
        else: # Not found
            print(url + ": No meta og:title")

            # Retrieve title from 'title' tag
            title = soup.find('title')
            if title != None:
                content += "Title: " + title.get_text() + "\n\n"
            else: # Not found
                print(url + ": No title tag")

        # Retrieve content only from 'main' tag
        info = soup.find('main')
        if info == None:
            # If class was not found, use entire doc
            print(url + ": No main tag found")
            info = soup

        # Scan info and retrieve "h1" and "p" divs
        for items in info.findChildren():
            #print("***items***")
            #print(items)

            # Following is not well formatted
            #items_text = items.get_text()
            #items_text = items_text.strip()
            #content += items_text

            # Scan children
            for item in items.find_all(["h1", "h2", "h3", "h4", "h5", "p"], recursive=False):
            #for item in items.find_all(recursive=False):
                # Remove 'style' and 'script'
                for el in item(['style', 'script']):
                    el.decompose()

                #print("<<item>>")
                #print(item)
                item_text = item.get_text()

                # Strip all whitespaces, tabs
                item_text = re.sub(r"[\t 　]*", "", item_text)
                # Append newline to each '。' and period
                item_text = re.sub(r"\r\n", "\n", item_text)
                #item_text = re.sub(r"\n\n", "\n", item_text)
                item_text = re.sub(r"\n+", "\n", item_text)
                #item_text = re.sub(r"。", "。\n", item_text)
                #item_text = re.sub(r".", ".\n", item_text)
                if len(item_text) > 0:
                    content += item_text + "\n"

        #print("**** Content ****\n" + content)

        # Create Document for this
        self.document = Document(
            page_content=content,
            metadata={"source": url}
        )

# Load HTMLs with PareHTML
class MyHTMLLoader:
    def __init__(
        self,
        urls
    ):
        # Access all websites and create list of Document
        self.documents = []
        for url in urls:
            self.documents.append(PareHTML(url).document)

        #print(self.documents)

    # Return self.documents
    def load(self):
        return self.documents

# Main (for debugging)
if __name__ == "__main__":
    #url = 'https://5hon-yubi.net/view/item/000000000883?category_page_id=ct314'
    #hp = PareHTML(url)
    #print(hp.document)

    urls = ['https://5hon-yubi.net/view/item/000000000883?category_page_id=ct314', 'https://5hon-yubi.net/view/category/ct302']

    html_data = MyHTMLLoader(urls)
    #print(html_data.documents)
    print(html_data.load())
