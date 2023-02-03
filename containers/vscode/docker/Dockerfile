FROM docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
	apt install -y software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y \
	curl \
	git \
	gcc \
	python3.10 \
	python3.10-dev \
	python3.10-venv

RUN mkdir /root/.virtualenvs && mkdir /root/.requirements

COPY requirements.txt /root/.requirements/feedback-prize/requirements.txt
RUN python3.10 -m venv /root/.virtualenvs/feedback-prize
RUN PATH="/root/.virtualenvs/feedback-prize/bin:$PATH" pip install -r /root/.requirements/feedback-prize/requirements.txt

# Install starship shell
RUN curl -sS https://starship.rs/install.sh > install_starship.sh && \
    chmod +x install_starship.sh && \
    ./install_starship.sh --yes \
    && echo "eval \"\$(starship init bash)\"" >> ~/.bashrc
