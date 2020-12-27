FROM registry.gitlab.com/tcl-research/internal/general/tcl-base-dev

USER tclbot
COPY requirements.txt /home/tclbot
WORKDIR /home/tclbot
# Install basic python stack
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt\
     --no-cache-dir

