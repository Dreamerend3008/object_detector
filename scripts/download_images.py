from icrawler.builtin import BingImageCrawler


categories = ['chair', 'kitchen knife', 'samsung phone']

for category in categories:
    bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': f'/home/harry/Github/object_detector/raw_images/{category}'})
    bing_crawler.crawl(keyword=category, max_num=100)
    print(f'Downloaded images for category: {category}')