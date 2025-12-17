pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        CLUSTER_NAME = "microdegree-cluster"
        NAMESPACE = "microdegree"
        DEPLOYMENT_NAME = "openai-chatbot"
        SERVICE_NAME = "openai-chatbot-service"
        DOCKER_USERNAME = "rashmidevops1"
        IMAGE_NAME = "${DOCKER_USERNAME}/genai-openai:${GIT_COMMIT}"
    }

    stages {
        stage('Git Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/rashmigmr13-eng/gen.git'
            }
        }

        stage('Build & Tag Docker Image') {
            steps {
                script {
                    sh 'printenv'
                    sh "docker build -t ${IMAGE_NAME} ."
                }
            }
        }

        stage('Docker Image Scan') {
            steps {
                script {
                    sh "trivy image --format table -o trivy-image-report.html ${IMAGE_NAME}"
                }
            }
        }

        stage('Login to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    sh "echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin"
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                sh "docker push ${IMAGE_NAME}"
            }
        }

        stage('Update EKS Config') {
            steps {
                sh "aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}"
            }
        }

        stage('Deploy to EKS') {
            steps {
                withKubeConfig(
                    caCertificate: '',
                    clusterName: "${CLUSTER_NAME}",
                    contextName: '',
                    credentialsId: 'kube',
                    namespace: "${NAMESPACE}",
                    restrictKubeConfigAccess: false,
                    serverUrl: 'https://CDDC6A719AA1EB6DB1B87808CBC3D2C2.gr7.us-east-1.eks.amazonaws.com'
                ) {
                    sh "sed -i 's|replace|${IMAGE_NAME}|g' deployment.yaml"
                    sh "kubectl apply -f deployment.yaml -n ${NAMESPACE}"
                    sh "kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}"
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                withKubeConfig(
                    caCertificate: '',
                    clusterName: "${CLUSTER_NAME}",
                    contextName: '',
                    credentialsId: 'kube',
                    namespace: "${NAMESPACE}",
                    restrictKubeConfigAccess: false,
                    serverUrl: 'https://CDDC6A719AA1EB6DB1B87808CBC3D2C2.gr7.us-east-1.eks.amazonaws.com'
                ) {
                    sh "kubectl get deployment -n ${NAMESPACE}"
                    sh "kubectl get pods -n ${NAMESPACE}"
                    sh "kubectl get svc -n ${NAMESPACE}"
                }
            }
        }
    }

    post {
        always {
            script {
                def jobName = env.JOB_NAME
                def buildNumber = env.BUILD_NUMBER
                def pipelineStatus = currentBuild.result ?: 'SUCCESS'
                def bannerColor = pipelineStatus.toUpperCase() == 'SUCCESS' ? 'green' : 'red'

                def body = """
                    <html>
                    <body>
                    <div style="border: 4px solid ${bannerColor}; padding: 10px;">
                        <h2>${jobName} - Build ${buildNumber}</h2>
                        <div style="background-color: ${bannerColor}; padding: 10px;">
                            <h3 style="color: white;">Pipeline Status: ${pipelineStatus.toUpperCase()}</h3>
                        </div>
                        <p>Check the <a href="${BUILD_URL}">console output</a>.</p>
                    </div>
                    </body>
                    </html>
                """

                emailext (
                    subject: "${jobName} - Build ${buildNumber} - ${pipelineStatus.toUpperCase()}",
                    body: body,
                    to: 'rashmigmr13@gmail.com',
                    from: 'rashmigmr13@gmail.com',
                    replyTo: 'rashmigmr13@gmail.com',
                    mimeType: 'text/html',
                    attachmentsPattern: 'trivy-image-report.html'
                )
            }
        }
    }
}
