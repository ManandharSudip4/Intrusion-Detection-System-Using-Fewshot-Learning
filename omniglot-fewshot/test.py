import torch
import time
from tqdm import tqdm
from model.ProtoNet import ProtoNet
from functions import read_images, extract_sample, extract_query


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        # print(output["y_hat"])
        running_loss += output["loss"]
        running_acc += output["acc"]
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("Test results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


if __name__ == "__main__":
    n_support = 5
    n_query = 5

   # print("Reading images...")
    t1 = time.time()
   # test_x, test_y = read_images(
   #     "/home/user/manualpartition/teamIDS/IDS/ai/IGTD-1dto2d/IGTD-results30Com/euclidean/data",
   #     n_support + n_query,
   # )

    test_x = torch.load(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_x_euclidean_one55.pt",
    )
    test_y = torch.load(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_y_euclidean_one55.pt",
    )

    t2 = time.time()

    print(len(test_y.unique()))

   # torch.save(
   #     test_x,
   #     "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_x_euclidean_one55.pt",
   # )
   # torch.save(
   #     test_y,
   #     "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_y_euclidean_one55.pt",
   # )

    print(
        f"Time required to read {test_y.size()} files: {t2-t1:.2f}s | {(t2-t1)/60:.2f}m"
    )

    n_way = len(test_y.unique())
    test_episode = 1000

    model = ProtoNet(
        x_dim=(1, 9, 9),
        hid_dim=16,
        z_dim=256,
    ).to("cuda:0")

    model.load_state_dict(
        torch.load(
            "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Models/70pCom_euclidean_one/model_12"
        )
    )

    model.eval()

    # test(model, test_x, test_y, n_way, n_support, n_query, test_episode)

    # TBD
    support = extract_sample(n_way, n_support, 0, test_x, test_y)
    print(support["sample"].size(), "------")
    query, cat = extract_query("/home/user/manualpartition/teamIDS/IDS/ai/IGTD1dto2d/IGTD-results30Com/euclidean/data")
    print(support["sample"].dtype, query.dtype)
    print(f"query: {query.size()}")
    y_hat = model.predict(support["sample"], query, 13, 5, 1)
    print("y_hat: ", y_hat, "cat: ", cat)
    # TBD

    value_of_n =y_hat.item()
    support_label = ['Infiltration', 'DoS Slowhttptest', 'SSH-Patator',
                            'Web Attack � Brute Force', 'FTP-Patator', 'DoS Hulk',
                            'Bot', 'Web Attack � XSS', 'DoS slowloris', 'DoS GoldenEye',
                            'DDoS', 'BENIGN', 'PortScan'
                            ]
    results = support_label[value_of_n]
    print("Result: ", results)

    # print(test_x.shape)
    # print(test_x.view(-1, 1, 9, 9).shape)
    # a = model(test_x.view(-1, 1, 9, 9)[:1, :, :, :].float().cuda())
    # # print('------------------------------------')
    # print(a)
    # print(a.size(-1))
