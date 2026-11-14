FROM python:3.10.8

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output \
    && chown user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/

ENV PATH="/home/user/.local/bin:${PATH}"
# COPY --chown=user:user vlm3d_cls/ /opt/app/
COPY --chown=user:user . /opt/app

RUN python -m pip install --user -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple && python -m pip install --user pip-tools -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m piptools sync requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "process.py" ]
