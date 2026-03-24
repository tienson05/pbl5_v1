import urllib.request
import urllib.parse

def send_locker(locker_id):
    # base_url = "http://192.168.101.55/open"
    base_url = "http://172.20.10.3/open"
    number = int(locker_id)
    params = urllib.parse.urlencode({
        "locker": number
    })

    url = f"{base_url}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            result = response.read().decode()
            print("Response:", result)

    except Exception as e:
        print("Error:", e)