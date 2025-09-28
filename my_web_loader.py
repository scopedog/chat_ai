import os
import sys
import requests
import re
from bs4 import BeautifulSoup
from loguru import logger
from langchain_core.documents import Document
from urllib.parse import urlparse

# Global parameters
Timeout = 7 # Connection timeout in sec for requests.get()

# Link info
class LinkInfo:
    def __init__(
        self,
        url: str = None,
        scan_content: bool = True
    ):
        # Initialize
        self.url = url
        self.scan_content = scan_content

# Pare HTML content
class PareHTML:
    def __init__(
        self,
        url: str = None,
        link_filter_keywords: list[str] = None,
        content_filter_keywords: list[str] = None
    ):
        # Initialize
        content = ""
        self.links = []
        self.doc = None
        #self.combined_ctx = "" # Combined context with no overlaps

        # For HTML, you shouldn't do this
        regex = []
        regex.append({"regex": re.compile(r"[ 　]+"), "replace": " "})
        '''
        regex.append({"regex": re.compile(r"[\t]+"), "replace": "\t"})
        regex.append({"regex": re.compile(r"\r\n"), "replace": "\n"})
        regex.append({"regex": re.compile(r"\n\n+"), "replace": "\n"})
        '''

        # Get base url
        parsed = urlparse(url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        #print("base_url: " + self.base_url)

        # Access website
        try:
            response = requests.get(url, timeout=Timeout)
        except requests.exceptions.Timeout:
            logger.warning("requests.get(" + url + ") timed out (" + \
                           str(Timeout) + "sec)")
            return
            #raise Exception("requests.get(" + url + ") timed out (" + \
            
        #page = response.text # Encoding fails with some sites
        page = response.content # This seems better, maybe...
        soup = BeautifulSoup(page, 'html.parser')

        # Retrieve title from 'meta property="og:title"'
        title = soup.find("meta", property="og:title")
        if title is not None:
            content += "Title: " + title["content"] + "\n\n"
        else: # Not found
            #print(url + ": No meta og:title")

            # Retrieve title from 'title' tag
            title = soup.find('title')
            if title is not None:
                content += "Title: " + title.get_text() + "\n\n"

        '''
        # Retrieve content only from 'main' tag
        info = soup.find('main')
        if info is None:
            # If class was not found, use entire doc
            print(url + ": No main tag found")
            info = soup
        '''
        info = soup
        #print(info.get_text(separator=' '))

        # Scan links and save to self.links
        for a in info.find_all(["a"], href=True):
            # Get link
            link = a['href']

            # Check link
            '''
            # This doesn't work because some use different URLs with same domain
            # Ex. https://timee.co.jp/, https://corp.timee.co.jp
            if (link.startswith('http://') or link.startswith('https://')) and \
               not link.startswith(self.base_url):
                # Different base URL  
                continue
            '''
            if link in self.links:
                continue

            a_content = None
            #print(a.contents)
            #print(len(a.contents))
            if len(a.contents) == 0:
                continue

            add_link = False
            for ac in a.contents:
                txt = ac.text
                if txt is None:
                    continue

                # Get content text
                a_content = txt.strip()
                #print("content: " + a_content)

                # Skip link not related to organization profile
                # This is only for ses-matching
                if a_content is None:
                    continue
                elif bool(link_filter_keywords):
                    # Check if a_content contains one of link_filter_keywords
                    add_link = any(w in a_content for w in link_filter_keywords)
                    '''
                    for k in link_filter_keywords:
                        if k in a_content:
                            add_link = True
                            break
                    '''

                    if add_link:
                        break

            # Check img alt
            if not add_link and link_filter_keywords is not None:
                for img in a.find_all("img", alt=True):
                    alt = img['alt']
                    add_link = any(w in alt for w in link_filter_keywords)
                    '''
                    for k in link_filter_keywords:
                        if k in alt:
                            add_link = True
                            break
                    '''

                    if add_link:
                        break

            if add_link:
                # Check link
                if not link.startswith('http'):
                    if link.startswith('/'):
                        link = link[1:]
                    if link.startswith('#'):
                        continue
                    link = link.strip()
                    if len(link) == 0: # Self URL
                        continue

                    link = self.base_url + "/" + link
                    #print("link: " + link)

                # We skip PDF
                if link.lower().endswith('.pdf'):
                    continue
                    
                # Append this link if not in list
                if link not in self.links:
                    self.links.append(link)

        # Retrieve tables
        for table in info.find_all(["table"]):
            #print("table: " + table.get_text())

            # Retrieve texts from each row 
            # We format tables in special way
            rows = table.find_all("tr")
            for row in rows:
                #print("row: " + row.get_text())

                # Find all cells (td or th)
                cells = row.find_all(["td", "th"])

                # Extract and clean text, join with tab so cells are
                # combined with tab
                values = [cell.get_text(strip=False) for cell in cells]
                #print(values)
                content += "\t".join(values) + "\n"
                #print("\t".join(values))

        # Remove some sections
        # Tables are already retrieved above
        for tag in info(['img', 'style', 'script', 'table']):
            tag.decompose()

        # Get entire text
        #content += info.get_text(separator=' ') # Do not do this...
        content += info.get_text()

        # Check if one of content_filter_keywords is included 
        # content_filter_keywords are examined with 'or'
        content_filter_passed = False
        if content_filter_keywords is not None:
            content_filter_passed = any(w in content for w in content_filter_keywords)
        else:
            content_filter_passed = True

        if not content_filter_passed:
            # Filtered out
            self.doc = None
            return

        # Reformat content
        for rgx in regex:
            content = rgx["regex"].sub(rgx["replace"], content)
        '''
        '''

        #print("**** Content ****\n" + content)

        # Create Document for this
        self.doc = Document(
                    page_content=content,
                    metadata={"source": url},
        )
        #print(self.doc)

# Load websites
class MyWebLoader:
    def __init__(
        self,
        urls: list[str],
        max_links=70,
    ):
        # Access all websites and create list of Document
        self.docs = []
        self.links = []
        self.combined_ctx = ""

        link_filter_keywords = \
            ["概要", "紹介", "会社情報", "企業情報", "案内", "問い合",
             "問合", "について", "about", "プロフィール"]
        content_filter_keywords = \
            ['〒', '住所', '所在地', '本社']

        for url in urls:
            try:
                html = PareHTML(url, link_filter_keywords=link_filter_keywords)
            except Exception as e:
                msg = "PareHTML: " + url + ": " + str(e)
                logger.error(msg)
                raise Exception(msg)
                return

            if html.doc is not None:
                self.docs.append(html.doc)

            #print(html.links)
            self.links = list(set(self.links + html.links))

        logger.debug("self.links: " + str(self.links))

        '''
        # Adjust links
        if len(self.links) > max_links:
            self.links = self.links[0:max_links]
        '''

        # Also scan links
        extra_links = []
        for url in self.links:
            html = PareHTML(url,
                            link_filter_keywords=link_filter_keywords,
                            content_filter_keywords=content_filter_keywords)
            extra_links = list(set(extra_links + html.links))
            doc = html.doc
            if doc is not None:
                self.docs.append(doc)
                self.combined_ctx += doc.page_content + "\n"

        # Get links that were not scanned yet
        extra_links = list(set(extra_links) - set(self.links))
        logger.debug("extra_links: " + str(extra_links))

        # Scan extra_links
        for url in extra_links:
            html = PareHTML(url,
                            link_filter_keywords=link_filter_keywords,
                            content_filter_keywords=content_filter_keywords)
            doc = html.doc
            if doc is not None:
                self.docs.append(doc)
                self.combined_ctx += doc.page_content + "\n"
        '''
        '''

        #print(self.docs)

    # Return self.docs
    def load(self):
        #print("self.docs: " + str(self.docs))
        return self.docs, self.links, self.combined_ctx

# Main (for debugging)
if __name__ == "__main__":
    urls = sys.argv[1:]

    html_data = MyWebLoader(urls)
    print(html_data.load())
