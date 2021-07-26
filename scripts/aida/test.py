import multiprocessing
import tqdm
import concurrent.futures
manager = multiprocessing.Manager()
shared_dict = manager.dict()
num = 10000


def worker1(pack):
	try:
		(key, v) = pack
		shared_dict[key] = v
	except:
		print(pack, 'error')

def threaded_work(l):
	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		executor.map(worker1, l)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]
        
l = [(i, i) for i in range(num)]
cl = chunks(l, 2)
with multiprocessing.Pool(8) as pool:
	# [ _ for _ in tqdm.tqdm(pool.imap_unordered(worker1, l), total=len(l))]
	[ _ for _ in tqdm.tqdm(pool.imap_unordered(threaded_work, cl), total = len(cl))]


print(shared_dict[999])