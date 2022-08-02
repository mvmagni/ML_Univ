# -*- coding: utf-8 -*-

# Sergei Bugrov
# 7-9-17
#
# Downloads all available books in English language in .txt format from http://www.gutenberg.org,
# unpacks them from .zip archives, saves them to ../books/ folder, and deletes .zip files.
#
# usage : python gutenberg.py
#
# python version : 3.6.1

import requests, bs4, os, errno, zipfile, glob
from urllib.request import urlretrieve


def main():

    if not os.path.exists('books/'):
        try:
            os.makedirs('books/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # STEP 1. BUILD A LIST OF URLS

    urls_to_books = []

    if not os.path.exists('urls_to_books.txt'):

        page_w_books_url = 'http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en'

        while 1 == 1:

            is_last_page = False

            print('Reading page: ' + page_w_books_url)

            page_w_books = requests.get(page_w_books_url, timeout=20.0)

            if page_w_books:
                page_w_books = bs4.BeautifulSoup(page_w_books.text, "lxml")
                urls = [el.get('href') for el in page_w_books.select('body > p > a[href^="http://aleph.gutenberg.org/"]')]
                url_to_next_page = page_w_books.find_all('a', string='Next Page')

                if len(urls) > 0:
                    urls_to_books.append(urls)

                    if url_to_next_page[0]:
                        page_w_books_url = "http://www.gutenberg.org/robot/" + url_to_next_page[0].get('href')
                else:
                    is_last_page = True

            if is_last_page:
                break

        urls_to_books = [item for sublist in urls_to_books for item in sublist]

        # Backing up the list of URLs
        with open('urls_to_books.txt', 'w') as output:
            for u in urls_to_books:
                output.write('%s\n' % u)

    # STEP 2. DOWNLOAD BOOKS

    # If, at some point, Step 2 is interrupted due to unforeseen
    # circumstances (power outage, lost of internet connection), replace the number
    # (value of the variable url_num) below with the one you will find in the logfile.log
    # Example
    #       logfile.log : Unzipping file #99 books/10020.zip
    #       the number  : 99
    url_num = 0

    if os.path.exists('urls_to_books.txt') and len(urls_to_books) == 0:
        with open('urls_to_books.txt', 'r') as f:
            urls_to_books = f.read().splitlines()

    for url in urls_to_books[url_num:]:

        dst = 'books/' + url.split('/')[-1].split('.')[0].split('-')[0]

        with open('logfile.log', 'w') as f:
            f.write('Unzipping file #' + str(url_num) + ' ' + dst + '.zip' + '\n')

        if len(glob.glob(dst + '*')) == 0:
            urlretrieve(url, dst + '.zip')

            with zipfile.ZipFile(dst + '.zip', "r") as zip_ref:
                try:
                    zip_ref.extractall("books/")
                    print(str(url_num) + ' ' + dst + '.zip ' + 'unzipped successfully!')
                except NotImplementedError:
                    print(str(url_num) + ' Cannot unzip file:', dst)

            os.remove(dst + '.zip')

        url_num += 1


if __name__ == '__main__':
    """
    The main function is called when gutenberg.py is run from the command line
    """

    main()