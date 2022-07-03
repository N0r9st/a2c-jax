import redis
import os, time
from warnings import warn
import pickle

class KLMJobServer:
    def __init__(self,
                 host="localhost",
                 port=7070,
                 jobs_key="jobs",
                 results_key="results",
                 negative_results_key="negative_results",
                 redis_init_wait_seconds=5,
                 password=None,
                 verbose=True,
                 base_prefix="",
                 **kwargs,
                 ):
        """
        The database instance that can stores paths, model weights and metadata.
        Implemented as a thin wrapper over Redis.
        :param host: redis hostname
        :param port: redis port
        :param args: args for database client (redis)
        :param kwargs: kwargs for database client (redis), e.g. password="***"
        :param default_prefix: prepended to both default_params_key and default_session_key and
            default_worker_prefix. Does NOT affect custom keys.
        :param jobs_key: default name for incomplete job
        :param results_key: default name for job results
        :param model_key: default name for weights pickle
        """
        self.jobs_key, self.results_key = (base_prefix + jobs_key), (base_prefix + results_key)
        self.verbose = verbose
        self.negative_results_key = base_prefix + negative_results_key

        # if localhost and can't find redis, start one
        if host in ("localhost", "0.0.0.0", "127.0.0.1", "*"):
            try:
                redis.Redis(host=host, port=port, password=password, **kwargs).client_list()
            except redis.ConnectionError:
                # if not, on localhost try launch new one
                if self.verbose:
                    print("Redis not found on %s:%s. Launching new redis..." % (host, port))
                self.start_redis(port=port, requirepass=password, **kwargs)
                time.sleep(redis_init_wait_seconds)

        self.redis = redis.Redis(host=host, port=port, password=password, **kwargs)
        if self.verbose and len(self.redis.keys()):
            warn("Found existing keys: {}".format(self.redis.keys()))

    def start_redis(self, **kwargs):
        """starts a redis serven in a NON-DAEMON mode"""
        kwargs_list = [
            "--{} {}".format(name, value)
            for name, value in kwargs.items()
        ]
        cmd = "nohup redis-server {} > .redis.log &".format(' '.join(kwargs_list))
        if self.verbose:
            print("CMD:", cmd)
        os.system(cmd)

    @staticmethod
    def dumps(data):
        """ converts whatever to string """
        return pickle.dumps(data, protocol=4)

    @staticmethod
    def loads(string):
        """ converts string to whatever was dumps'ed in it """
        return pickle.loads(string)

    def reset_queue(self, prefix=""):
        for key in self.jobs_key, self.results_key, self.negative_results_key:
            if prefix:
                key = prefix + "::" + key
            self.redis.delete(key)
        return True
    
    def add_jobs(self, *jobs):
        return self.redis.rpush(self.jobs_key, *map(self.dumps, jobs))

    def commit_result(self, result, negative=False, prefix=""):
        if negative: key = self.negative_results_key
        else: key = self.results_key
        if prefix:
            key = prefix + "::" + key
        return self.redis.rpush(key, self.dumps(result))

    def get_result(self, timeout=0, negative=False, prefix=""):
        if negative: key = self.negative_results_key
        else: key = self.results_key
        if prefix:
            key = prefix + "::" + key
        payload = self.redis.blpop(key, timeout=timeout)
        if payload is not None: return self.loads(payload[1])

    def get_job(self, timeout=0):
        payload = self.redis.blpop(self.jobs_key, timeout=timeout)
        return self.loads(payload[1])

    def get_job_results(self, current_iteration, num_expected_jobs, timeout=30, negative=False, prefix=""):
        st = time.time()
        results = [self.get_result(negative=negative, prefix=prefix)]
        wtd = time.time() - st
        times = [wtd]
        
        incorrect_iteration = 0
        while True:
            st = time.time()
            result = self.get_result(timeout=timeout, negative=negative, prefix=prefix)
            wtd = time.time() - st
            times.append(wtd)
            if result is None: break
                
            if result['iteration']==current_iteration: 
                results.append(result)
            else:
                incorrect_iteration += 1
                
            if len(results) == num_expected_jobs: break
                
        logs = dict(
            n_results=len(results),
            n_latecomers=incorrect_iteration,
            time_spent=sum(times),
            mean_time=sum(times)/len(times),
            max_time=max(times),
            min_time=min(times)
        )
        
        return results, logs