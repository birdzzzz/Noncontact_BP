
import time, queue
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
# 任务个数
task_number = 1;
# 定义收发队列
task_queue = queue.Queue(task_number);
result_queue = queue.Queue(task_number);
print('task_queue.qsize() = ',task_queue.qsize())
def gettask():
    return task_queue;


def getresult():
    return result_queue;

def getConvResult():
    return ConvResult_queue;
def test():
    # windows下绑定调用接口不能使用lambda，所以只能先定义函数再绑定；把两个Queue都注册到网络上, callable参数关联了Queue对象:
    BaseManager.register('get_task', callable=gettask);
    BaseManager.register('get_result', callable=getresult);
    BaseManager.register('get_ConvResult', callable=getConvResult);
    # 绑定端口并设置验证码，windows下需要填写ip地址，linux下不填默认为本地
    manager = BaseManager(address=('114.213.234.74', 8786), authkey=b'123');
    # 启动
    manager.start();
    try:
        # 通过网络获取任务队列和结果队列
        task = manager.get_task();
        print('task.qsize():',task.qsize())   # 0
        result = manager.get_result();
        i= loaddata1("F:/data")
        task.put(i);
        # 每秒检测一次是否所有任务都被执行完
        print('task.qsize():',task.qsize())  # 10
        print('result.full():',result.full())
        while not result.full():
            time.sleep(0.5);
        for i in range(result.qsize()):
            ans = result.get();
            print('task %d is finish , runtime:%d s' % ans);
    except:
        print('Manager error');
    finally:
        # 一定要关闭，否则会爆管道未关闭的错误
        manager.shutdown();

if __name__ == '__main__':
    # windows下多进程可能会炸，添加这句可以缓解
    freeze_support()
    test();