<?php
session_start();
$conn = new mysqli("localhost", "root", "", "website_login");

if ($conn->connect_error) {
    die(json_encode(["error" => "Database connection failed"]));
}

if (!isset($_SESSION['users'])) {
    die(json_encode(["error" => "User not logged in"]));
}

$username = $_SESSION['users'];
$sql = "SELECT full_name, username, email, age, gender, bio, profile_picture FROM users WHERE username=?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $username);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows === 1) {
    echo json_encode($result->fetch_assoc());
} else {
    echo json_encode(["error" => "User not found"]);
}

$stmt->close();
$conn->close();
?>
