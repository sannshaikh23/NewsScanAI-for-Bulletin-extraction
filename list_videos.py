from db_utils import get_all_videos

videos = get_all_videos()
print(f'Total videos: {len(videos)}\n')
print('All video names:')
for v in videos:
    name = v['name']
    show_in_samples = v['show_in_samples']
    is_indexed = v['is_indexed']
    print(f'  - {name} (sample: {show_in_samples}, indexed: {is_indexed})')
