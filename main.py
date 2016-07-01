import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from time import time
import os

from multiprocessing import Pool
import html2text


import click


def parse_post(url='http://ailev.livejournal.com/1274596.html'):
    r = requests.get(url)
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, 'html.parser')
    try:
        title = soup.find('h1', class_='entry-title').text.strip()
    except:
        title = ''
    h = html2text.HTML2Text()
    try:
        content = soup.find('article', class_='entry-content').text.strip('\n ')
        cleared = h.handle(content).strip('\n ')
    except:
        cleared = ''
    if len(title) == 0 and len(cleared) == 0:
        title = '#FAILED'
    return (url, title, cleared)


def list_days(root_url='http://ailev.livejournal.com/'):
    """
    :param root_url:
    :return:
     list of links like `user.livejournal.com/year/month/day/`
    """
    hrefs = []
    for year in tqdm(range(2002, 2017)):
        calendar = root_url + '{}/'.format(year)
        r = requests.get(calendar)
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        for x in soup.findAll('a'):
            link = x.get('href', '')
            if len(link) >= len(root_url) and link[:len(root_url)] == root_url:
                poi = link[len(root_url):]
                if re.match('(\d+)\/(\d+)\/(\d+)/', poi):
                    hrefs.append(link)
    return hrefs


def list_posts(day_link):
    hrefs = []

    # just dirty filtering trick
    root = day_link[:15]
    r = requests.get(day_link)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    for x in soup.findAll('a'):
        link = x.get('href', '')
        if re.match('(.+)\/(\d+).html$', link) and link[:len(root)] == root:
            hrefs.append(link)
    return hrefs


def all_links(root='http://ailev.livejournal.com/', nb=10, path='post-list.txt'):
    print('Fetch calendar entries')
    days = list_days(root_url=root)
    print('There are {} days with entries'.format(len(days)))
    t0 = time()

    pool = Pool(processes=nb)
    it = pool.imap_unordered(list_posts, days)
    work = list(tqdm(it, total=len(days)))
    pool.close()
    pool.join()

    links = []
    for x in work:
        if x:
            links.extend(x)

    links = list(set(links))
    links.sort()
    

    with open(path, 'w') as fout:
        fout.writelines(x + '\n' for x in links)

    t1 = time()
    print('Done for {}s'.format(t1 - t0))
    return links


def fetch_list(links, nb=4):
    print('Going to mp load all {} pages'.format(len(links)))
    t0 = time()
    pool = Pool(processes=nb)
    it = pool.imap_unordered(parse_post, links)
    work = list(tqdm(it, total=len(links)))
    pool.close()
    pool.join()
    t1 = time()
    print('Done for {:.2f} minutes'.format((t1 - t0)/60))
    work = [_ for _ in work if _]
    work.sort(key=lambda t: t[0])

    return work


@click.command()
@click.option('--nb', default=4, help='number of workers')
@click.option('--save-dir', default='tmp/')
def main(nb, save_dir):
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    links = all_links(nb=nb, path=os.path.join(save_dir, 'post-list.txt'))

    data = fetch_list(links, nb=nb)
    print('Store data')
    with open(os.path.join(save_dir, 'allj.txt'), 'w') as fout:
        for (url, title, content) in tqdm(data):
            fout.write('<post href="{}">{}\n'.format(url, title))
            fout.write(content)
            fout.write('\n</post>\n')


if __name__ == '__main__':
    main()
