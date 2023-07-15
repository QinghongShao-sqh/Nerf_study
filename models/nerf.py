import  torch
from  torch import  nn

class Embedding(nn.Module):
    def __init__(self, in_channels,N_freqs,logscale=True):
            # 定义一个函数，将x嵌入到(x, sin(2 ^ k x)， cos(2 ^ k x)，…)
            # In_channels: 输入通道的数量(xyz和方向都是3)
        super(Embedding,self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels* (len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2** torch.linspace(0,N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1,2**(N_freqs-1),N_freqs)

    def forward(self,x):
        # ”“”
        # 将x嵌入到(x, sin(2 ^ k x) cos(2 ^ k x)，…)  与论文不同的是，“x”也在输出中
        # 参见https: // github.com / bmild / nerf / issues / 12
        # 输入:
        # x: (B, self.in_channels)
        # 输出:
        # out: (B, self.out_channels)
        # ”“”
        out =[x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out+= [func(freq*x)]
        return  torch.cat(out,-1)


class NeRF(nn.Module):
    def __init__(self,D=8,W=256,in_channels_xyz=63, in_channels_dir=27,skips=[4]):
        """
        D:  编码器的层数
        W: 每层隐藏单位的数量
        In_channels_xyz: xyz的输入通道数(默认为3 + 3 * 10 * 2 = 63)
        In_channels_dir: 方向的输入通道数(默认为3 + 3 * 4 * 2 = 27)
        skip: 在第d层增加skip连接
        """
        super(NeRF, self).__init__()
        self.D =D
        self.W =W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips= skips

        # xyz 编码层
        for i in range(D):
            if i==0:
                layer = nn.Linear(in_channels_xyz,W)
            elif i in skips:
                layer = nn.Linear(W+ in_channels_xyz,W)
            else:
                layer = nn.Linear(W,W)
            layer = nn.Sequential(layer,nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1},layer")

        self.xyz_encoding_final = nn.Linear(W,W)

        #  对输入的视角向量进行编码的层
        self.dir_encoding - nn.Sequential(
            nn.Linear(W+ in_channels_dir,W // 2),
            nn.ReLU(True)
        )
        # output  layers
        self.sigma = nn.Linear(W,1)
        self.rgb = nn.Sequential(
            nn.Linear(W//2 ,3),
            nn.Sigmoid()
        )

    def forward(self,x,sigma_only=False):
        """
                将输入(xyz + dir)
                编码为rgb + sigma(尚未准备好渲染)。
                要渲染该光线，请参阅rendering.py
                输入:
                (B, self.in_channels_xyz(+self.in_channels_dir))
                位置和方向的嵌入向量
                Sigma_only: 是否只推断sigma。如果这是真的,
                x的形状是(B, self.in_channels_xyz)
                输出:
                如果sigma_ony:
                (B, 1)
                其他:
                out: (B, 4)， rgb和sigma
               """
        if not sigma_only:
            input_xyz , input_dir =\
                torch.split(x,[self.in_channels_xyz, self.in_channels_dir], dim=-1 )
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if  i in self.skips:
                xyz_ = torch.cat([input_xyz,xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return  sigma
        #从这里也额可以看出来，输出的密度只受xyz 位置影响， 颜色受xyz，以及dir 的共同影响
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final,input_dir],-1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb =self.rgb(dir_encoding)
        out = torch.cat([rgb, sigma],-1)
        return  out



















