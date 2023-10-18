from bs4 import BeautifulSoup
import requests


url = "https://infoselection.ru/infokatalog/literatura-knigi/literatura-obshchee/item/885-150-zolotykh-fraz-iz-russkoj-literatury"

def main():
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    text = soup.find("div", class_="dopol1")
    phrase_array = text.find_all("li")
    with open("RNN\phrase.txt", "w", encoding="utf-8-sig") as f:
        for phrase in phrase_array:
            f.write(phrase.text + "\n")

if __name__ == "__main__":
    main()