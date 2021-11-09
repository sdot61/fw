# Copy and paste the contents of this file into the AWS Lightsail Console
# Launch Script input when creating a new Instance
# Operations:
#   - installs AWS CodeDeploy Client and Reqs
#   - installs Python3.8
#   - installs Pip for Python 3.x

mkdir -p /etc/codedeploy-agent/conf

cat <<EOT >> /etc/codedeploy-agent/conf/codedeploy.onpremises.yml
---

aws_access_key_id: AKIAW24L2O7NAH6UYZNJ

aws_secret_access_key: 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

iam_user_arn: arn:aws:iam::470055286746:user/GithubActionsUser

region: us-east-1

EOT

yum update
yum install -y wget ruby

wget https://aws-codedeploy-us-west-2.s3.us-west-2.amazonaws.com/latest/install

chmod +x ./install

env AWS_REGION=us-east-1 ./install rpm

amazon-linux-extras enable python3.8
yum install python3.8 -y
yum install python3-pip -y
