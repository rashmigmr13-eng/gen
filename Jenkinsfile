pipeline {
    agent any

    environment {
        AWS_REGION       = "us-east-1"
        CLUSTER_NAME     = "microdegree-cluster"
        NAMESPACE        = "microdegree"
        DEPLOYMENT_NAME  = "openai-chatbot"
        SERVICE_NAME     = "openai-chatbot-service"
        DOCKER_USERNAME  = "rashmidevops1"
        IMAGE_REPO       = "${DOCKER_USERNAME}/genai-openai"
    }

    stages {
        stage('Git Checkout') {
            steps { checkout scm }
        }

        stage('Build & Tag Docker Image') {
            steps {
                script {
                    env.IMAGE_TAG  = env.GIT_COMMIT.take(7)
                    env.IMAGE_NAME = "${IMAGE_REPO}:${IMAGE_TAG}"
                    sh "docker build -t ${IMAGE_NAME} ."
                }
            }
        }

        stage('Scan Docker Image') {
            steps {
                sh "trivy image --format table -o trivy-image-report.html ${IMAGE_NAME}"
            }
        }

        stage('Login to Docker Hub') {
            steps {
                withCredentials([
                    usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')
                ]) {
                    sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                }
            }
        }

        stage('Push Docker Image') {
            steps { sh "docker push ${IMAGE_NAME}" }
        }

        stage('Update EKS Config') {
            steps {
                sh "aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}"
            }
        }

        stage('Deploy to EKS') {
            steps {
                withKubeConfig(credentialsId: 'kube', clusterName: "${CLUSTER_NAME}", namespace: "${NAMESPACE}") {
                    sh """
                        sed -i 's|replace|${IMAGE_NAME}|g' deployment.yaml
                        kubectl apply -f deployment.yaml -n ${NAMESPACE}
                        kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}
                    """
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                withKubeConfig(credentialsId: 'kube', clusterName: "${CLUSTER_NAME}", namespace: "${NAMESPACE}") {
                    sh """
                        kubectl get deploy -n ${NAMESPACE}
                        kubectl get pods -n ${NAMESPACE}
                        kubectl get svc -n ${NAMESPACE}
                    """
                }
            }
        }
    }

    post {
        always {
            emailext(
                subject: "${JOB_NAME} #${BUILD_NUMBER} - ${currentBuild.currentResult}",
                body: """
                <h3>Pipeline Status: ${currentBuild.currentResult}</h3>
                <p>Job: ${JOB_NAME}</p>
                <p>Build: ${BUILD_NUMBER}</p>
                <p><a href="${BUILD_URL}">View Console Output</a></p>
                """,
                to: 'rashmigmr13@gmail.com',
                mimeType: 'text/html',
                attachmentsPattern: 'trivy-image-report.html'
            )
        }
    }
}
