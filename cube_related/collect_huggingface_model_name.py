import requests
from bs4 import BeautifulSoup
import os
import time

class ModelCard:
    def __init__(self, model_name, task_name, download, likes_num):
        self.model_name = model_name
        self.task_name = task_name
        self.download = download
        self.likes_num = likes_num

    def __str__(self):
        return f"[model_name: {self.model_name}, task_name: {self.task_name}, download: {self.download}, likes_num: {self.likes_num}]"

def get_text_following_svg(soups, class_name):  
    svg_element = soups.find('svg', class_=class_name)  
    if svg_element and svg_element.next_sibling:  
        return svg_element.next_sibling.strip()  
    return None


if __name__ == "__main__":
    url = 'https://huggingface.co/models?p={page_num}&sort=likes'
    target_model_num = 100
    target_task = "Natural Language Processing"
    
    # get needed sub tasks
    page = requests.get(url.format(page_num=0))
    soup = BeautifulSoup(page.content, 'html.parser')
    task_task = soup.find('div', class_='mb-20 lg:mb-4')
    tasks = task_task.select('div[class="mb-3"]')
    tasks_dict = {}
    for task in tasks:
        task_name = task.find('div', class_="mb-3 text-sm font-medium text-gray-500").text.strip()
        sub_tasks = task.find_all('a', class_='tag-white')
        sub_task_names = []
        for sub_task in sub_tasks:
            sub_task_name = sub_task.find("span").text.strip()
            sub_task_names.append(sub_task_name)
        tasks_dict[task_name] = sub_task_names
    print(f"{target_task}: {tasks_dict[target_task]}")

    # get model cards
    model_cards = []
    model_name_set = set()
    page_num = 0
    while len(model_cards) < target_model_num:
        time.sleep(1)
        page = requests.get(url.format(page_num=page_num))
        soup = BeautifulSoup(page.content, 'html.parser')
        models = soup.find_all('article', class_='overview-card-wrapper')
        for model in models:
            model_name = model.find('h4', class_='text-md').text
            infos = model.find('div', class_='items-center')
            task_name = get_text_following_svg(infos, "mr-1.5 text-[.8rem]")
            download = get_text_following_svg(infos, "flex-none w-3 text-gray-400 mr-0.5")
            likes_num = get_text_following_svg(infos, "flex-none w-3 text-gray-400 mr-1")
            if task_name in tasks_dict[target_task] and model_name not in model_name_set:
                model_card = ModelCard(model_name, task_name, download, likes_num)
                model_cards.append(model_card)
                model_name_set.add(model_card.model_name)
                print(f"Percentage: {len(model_cards)/target_model_num*100:.2f}%, Process: {len(model_cards)}/{target_model_num}, {model_card}")
                if (len(model_cards) >= target_model_num):
                    break
        page_num += 1
    # for model in model_cards:
    #     print(model)
    
    # write to file
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    if not os.path.exists(f"{current_folder}/models"):
        os.makedirs(f"{current_folder}/models")
    with open(f"{current_folder}/models/{target_task}", "w+") as f:
        f.write('\n'.join(model.model_name for model in model_cards))
