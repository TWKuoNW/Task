import threading
import subprocess

def run_code_in_thread1():
    print("Starting thread1")
    subprocess.run(["python", "smart_fence_V1_1.py"])
    print("Ending thread1")

def run_code_in_thread2():
    print("Starting thread2")
    subprocess.run(["python", "smart_fence_V1_2.py"])
    print("Ending thread2")

def run_code_in_thread3():
    print("Starting thread3")
    subprocess.run(["python", "smart_fence_V1_3.py"])
    print("Ending thread3")

def run_code_in_thread4():
    print("Starting thread4")
    subprocess.run(["python", "smart_fence_V1_4.py"])
    print("Ending thread4")

def run_code_in_thread5():
    print("Starting thread5")
    subprocess.run(["python", "smart_fence_V1_5.py"])
    print("Ending thread5")


def run_code_in_thread6():
    print("Starting thread6")
    subprocess.run(["python", "smart_fence_V2_1.py"])
    print("Ending thread6")

def run_code_in_thread7():
    print("Starting thread7")
    subprocess.run(["python", "smart_fence_V2_2.py"])
    print("Ending thread7")

def run_code_in_thread8():
    print("Starting thread8")
    subprocess.run(["python", "smart_fence_V2_3.py"])
    print("Ending thread8")

def run_code_in_thread9():
    print("Starting thread9")
    subprocess.run(["python", "smart_fence_V2_4.py"])
    print("Ending thread9")

def run_code_in_thread10():
    print("Starting thread10")
    subprocess.run(["python", "smart_fence_V2_5.py"])
    print("Ending thread10")


# 創建並啟動執行緒
thread1 = threading.Thread(target=run_code_in_thread1)
thread1.start()
thread2 = threading.Thread(target=run_code_in_thread2)
thread2.start()
thread3 = threading.Thread(target=run_code_in_thread3)
thread3.start()
thread4 = threading.Thread(target=run_code_in_thread4)
thread4.start()
thread5 = threading.Thread(target=run_code_in_thread5)
thread5.start()
thread6 = threading.Thread(target=run_code_in_thread6)
thread6.start()
thread7 = threading.Thread(target=run_code_in_thread7)
thread7.start()
thread8 = threading.Thread(target=run_code_in_thread8)
thread8.start()
thread9 = threading.Thread(target=run_code_in_thread9)
thread9.start()
thread10 = threading.Thread(target=run_code_in_thread10)
thread10.start()

# 等待子執行緒結束
thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()
thread6.join()
thread7.join()
thread8.join()
thread9.join()
thread10.join()
