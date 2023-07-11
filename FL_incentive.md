联邦学习奖励机制，该机制为服务器根据客户端贡献发送对应奖励。
贡献是通过Shapley值计算出该客户端的梯度对服务器的更新起到多大的作用。奖励是服务器发送给客户端下次训练时附加的梯度。
1.服务器，发布上一轮结果Wt（全局参数和奖励，第一轮训练之前，客户端接受的奖励是0）。服务器发布一个较大且均匀分布的密钥池，含有密钥和公钥。
1.1 客户端总数设置为60000，随机挑选k个客户端形成一组，默认k为10000
2.客户端接受上一轮服务器发送的全局参数和自身对应奖励进行训练得到Xi，进行privatization mechanism对梯度进行保护。
2.1客户端i从服务器的密钥池中随机抽取私钥sk_i并生成对应公钥pk_i,Xi通过privatization mechanism（laplace mechanism和梯度裁剪，每个客户端裁剪比例𝐶= 1/100。）得出数据Xii的版本记为Ti1。
2.2客户端通过服务器的公钥(pk_c,Ti1)加密获得Xiii=Encoder(pk_c,(pk_i,Xii))
2.3客户端为消息计算一个签名ti=Sign(sk_i,Xiii),并发送给Shuffler
3.Shuffler收到1个用户组的信息{ti,Xiii}随机排列发送给服务器。{在该奖励机制中只有一个用户组,组内可多个客户端}////一对组时是{ti,Xiii}{ti,Xjjj}
4，服务器接受并解密{ti,Xiii},使用服务器私钥解密tii来获得(pk_i,ti)=decoder(sk_c,tii)
4.1服务器解密获得梯度信息后，进行平均梯度后梯度下降，
4.2服务器通过shapley值计算出各个客户端提供的梯度所对应的贡献值有多少，发送多少的奖励。
4.3各个客户客户端贡献相同，奖励相同。贡献更多，奖励更多。提过虚假贡献(梯度)的恶意客户端，将收到惩罚奖励，或者排除在外，不提供梯度奖励。
5.每个客户端从服务器下载并解密反向信息（从服务器发放的对应的梯度奖励）
重复以上流程 直至收敛。

数据集：Mnist数据集
模型：SampleConvNet 客户端和服务器都是用SampleConvNet模型，都是用随机梯度下降。
评估指标：MSE损失函数，running time result
加密算法：使用Paillier加密来执行安全的向后(梯度)信息传递
调用from opacus import PrivacyEngine统计隐私消耗
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

使用pytorch框架。

被挑选的客户端初始梯度：如果参与了上一轮的训练，则使用上一轮的贡献值梯度作为训练起始点，如果没参加上一轮，使用全局平均梯度进行训练，或者无用，奖励为0
{换句话说，激励更多客户端参与训练，也设计一种合理的回报留住更多的客户端}
#Collaborative Machine Learning with Incentive-Aware Model Rewards
{
    1.基于信誉的shapley值贡献反馈的联邦学习框架  //初始信誉，上传梯度给予的模型质量，多样性，历史信誉。//暂时以acc/mse为考核标准 -->每轮的梯度的贡献比例
    2.基于货币奖励的shapely值贡献反馈的联邦学习框架 //
}







实验包含：   数值结果/效率结果
1，是否含有非对称加密算法 
2，cdp/ldp所估计的梯度误差  test_acc
3，cdp/ldp 与 真实梯度的shapley值对比 