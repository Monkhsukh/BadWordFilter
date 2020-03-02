FROM python:3
COPY ./Profanity /Profanity
WORKDIR /Profanity
RUN pip install requests
RUN pip install numpy
RUN pip install Cython 
RUN pip install sklearn
RUN pip install requests
ENTRYPOINT ["setURL.sh"]
CMD ["python", "./server_profanity.py"]
