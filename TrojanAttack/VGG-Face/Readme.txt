1. 首先运行debug_all_layer_delta_trigger.py，该程序会加载Neural Cleanse逆向的trigger和100张clean image，遍历模型的每一层找到和后门相关的神经元，并将结果保存在mask_back和txt_file目录下。

2. 然后运行debug_test_flip.py或者debug_test_mute.py来修复后门。debug_test_flip.py对上一步找到的后门相关神经元的连接权重进行翻转，而debug_test_flip.py对上一步找到的后门相关神经元的连接权重置零。

3. 用TrojanNN的两个开源模型进行实验。权重翻转将模型square的后门攻击成功率降低到0.0%，干净数据准确率维持在97.01%，将模型watermark的后门攻击成功率降低到0.0%，干净数据准确率维持在97.02%。