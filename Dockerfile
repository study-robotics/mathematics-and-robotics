FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG ja_JP.UTF-8
RUN apt-get update && apt-get install -y \
    sudo \
    python3 python3-pip \
    git \
    language-pack-ja \
    vim \
    x11-apps \
    curl \
    zip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip3 install \
    # numpy:主に，行列計算に使用
    numpy \
    # pandas:データ分析
    pandas \
    # matplotlib:描画
    matplotlib \
    # openCV:画像処理
    opencv-python opencv-contrib-python \
    # scikit-learn:簡易的な機械学習ライブラリ
    scikit-learn

# そのままだと，matplotlibのplt.show()でエラーが起きるので，python3-tkをインストール
RUN apt-get update && apt-get install -y \
    python3-tk

# https://www.idnet.co.jp/column/page_187.html
# JupyterLab関連のパッケージ（いくつかの拡張機能を含む）
RUN python3 -m pip install --upgrade pip \
&&  pip install --no-cache-dir \
    black \
    jupyterlab \
    jupyterlab_code_formatter \
    jupyterlab-git \
    lckr-jupyterlab-variableinspector \
    jupyterlab_widgets \
    ipywidgets \
    import-ipynb

RUN python3 -m pip install --upgrade pip \
&& pip3 install sympy

# animationに必要
RUN python3 -m pip install --upgrade pip \
&& pip3 install menpo ffmpeg ffmpeg-python

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# rootでログインすると，全部のファイルがroot権限になって扱いが面倒なので，ユーザを作成
ARG DOCKER_UID=1000
ARG DOCKER_USER=docker
RUN useradd -m -s /bin/bash --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
    && echo $DOCKER_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DOCKER_USER \
    && chmod 0440 /etc/sudoers.d/$DOCKER_USER
# 作成したユーザーに切り替える
USER ${DOCKER_USER}
# -----------------------------------------
# host側と通信(GUIアプリを表示できるように)---------------
# WSL2 or Macのときのみ ubuntuの時は下記をコメントアウト
ENV DISPLAY=host.docker.internal:0
# -----------------------------------------------------
WORKDIR /workspace

CMD ["/bin/bash"]