import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm


class FreepikCrawler:
    def __init__(self, save_repo, base_url):
        self.save_repo = save_repo
        self.save_class = None
        self.base_url = base_url

    @classmethod
    def getUrl(cls, url):
        """Request the URL and return its HTML content."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, verify=False)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to fetch page: {url}, error: {e}")
            return None

    def get_image_links_from_page(self, html):
        """Extract image links from a single page."""
        soup = BeautifulSoup(html, "html.parser")
        all_img = soup.find_all('img')
        image_links = []
        for img in all_img:
            img_url = img.get('src')  # 改为获取 'src' 属性
            if img_url:
                image_links.append(img_url)
        return image_links

    def get_image_links_from_search(self, search_term, max_pages):
        """Collect image links from multiple search result pages."""
        print('Collecting image links...')
        all_image_links = []
        base_url = f'{self.base_url}/{search_term}'

        for page in range(1, max_pages + 1):
            if page == 1:
                page_url = base_url
            else:
                page_url = f"{base_url}/{page}"

            page_html = self.getUrl(page_url)
            if page_html:
                image_links = self.get_image_links_from_page(page_html)
                all_image_links.extend(image_links)
                print(len(image_links))
            else:
                break

        print(f'Collected {len(all_image_links)} image links.')
        return all_image_links

    def download_images(self, search_term, max_pages):
        """Download images from Freepik based on the search term."""
        self.save_class = os.path.join(self.save_repo, search_term)
        if not os.path.exists(self.save_class):
            os.mkdir(self.save_class)

        image_links = self.get_image_links_from_search(search_term, max_pages)
        success_len = 0
        new_len = 0

        print(f'Downloading {len(image_links)} images...')
        for img_url in tqdm(image_links):
            try:
                image_name = os.path.basename(img_url.split('?')[0])
                image_path = os.path.join(self.save_class, image_name)
                if not os.path.exists(image_path):
                    response = requests.get(img_url)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    new_len += 1
                success_len += 1
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

        print(f'Download complete: {new_len} new images saved to {self.save_class}, {success_len} images processed.')


if __name__ == '__main__':
    search_term = 'poster'  # The term for searching
    your_repo_path = '/Users/wangruihan/Documents/本科课程/机器学习/project/data'  # Set the directory where you want to save the images
    base_url = 'https://www.freepik.com/free-photos-vectors'  # Set the net url for crawl
    crawler = FreepikCrawler(your_repo_path, base_url)
    crawler.download_images(search_term, max_pages=70)  # Set the number of pages to scrape