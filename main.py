import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from time import time
import os

from multiprocessing import Pool


import click


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
    work = list(tqdm(it))
    pool.close()
    pool.join()

    print(work)
    links = []
    for x in work:
        if x:
            links.extend(x)
    print(links)

    with open(path, 'w') as fout:
        fout.writelines(x + '\n' for x in sorted(links))

    t1 = time()
    print('Done for {}s'.format(t1 - t0))


@click.command()
@click.option('--nb', default=4, help='number of workers')
@click.option('--save-dir', default='tmp/')
def main(nb, save_dir):
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    all_links(nb=nb, path=os.path.join(save_dir, 'post-list.txt'))


if __name__ == '__main__':
    main()

