import smtplib
from email.mime.text import MIMEText
from email.header import Header
import os
def send_email(content, header):
    # print(header)
    # print(content)
    # return
    try:
        with open(os.path.join(os.path.expanduser('~'),'.email_setting')) as f:
            lines = f.readlines()
            from_addr = lines[0].strip()
            password = lines[1].strip()
            smtp_server = lines[2].strip()
            to_addr = lines[3].strip()

        str_msg = content
        msg = MIMEText(str_msg, 'plain', 'utf-8')
        msg['From'] = Header(from_addr)
        msg['To'] = Header(to_addr)
        msg['Subject'] = Header(header)
        server = smtplib.SMTP_SSL(host=smtp_server)
        server.connect(smtp_server, 465)
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, msg.as_string())
        server.quit()
    except:
        print('send email failed!')
